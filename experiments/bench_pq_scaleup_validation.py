"""Scale-up validation of the "PQ without RHT beats snapvec" finding.

Before we turn this into a feature, confirm the 15-pp claim is
robust across:

* **K**         — ``K ∈ {16, 64, 256}`` (16 was our sweep default;
                  FAISS typically uses 256 = 8 bits/subspace).
* **Seed**      — each config repeated with 3 seeds, mean ± std.
* **Train/eval split** — codebooks trained on a 2 000-vec subset;
                  index built over the full 3 000 corpus;
                  300 disjoint queries.
* **Storage**   — explicit tight-packed B/vec accounting for all
                  four baselines (snapvec b=2,3,4 and PQ variants).

If PQ-noRHT still clears snapvec b=3/b=4 at matched or lower storage
with tight bounds on the std, we ship it.  If the gap closes once K
grows (because PQ-RHT also improves), the story is subtler and we
document that.

Run: ``python experiments/bench_pq_scaleup_validation.py``
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from snapvec import SnapIndex
from snapvec._rotation import padded_dim, rht


# ──────────────────────────────────────────────────────────────────── #
# K-means                                                               #
# ──────────────────────────────────────────────────────────────────── #

def _kpp(X: NDArray[np.float32], K: int, rng: np.random.Generator) -> NDArray[np.float32]:
    n = X.shape[0]
    centers = [X[int(rng.integers(n))]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        probs = d2 / (d2.sum() + 1e-12)
        nxt = int(rng.choice(n, p=probs))
        centers.append(X[nxt])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    return np.stack(centers).astype(np.float32)


def kmeans_mse(
    X: NDArray[np.float32], K: int, n_iters: int = 15, seed: int = 0,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    C = _kpp(X, K, rng)
    for _ in range(n_iters):
        # Chunked assignment to bound memory at K=256, N=2000, d_sub=16.
        d2 = (
            (X ** 2).sum(1, keepdims=True)
            - 2 * X @ C.T
            + (C ** 2).sum(1)[None, :]
        )
        asn = d2.argmin(1)
        newC = np.empty_like(C)
        for k in range(K):
            m = asn == k
            newC[k] = X[m].mean(0) if m.any() else X[d2.min(1).argmax()]
        if np.allclose(newC, C, atol=1e-5):
            return newC
        C = newC
    return C


# ──────────────────────────────────────────────────────────────────── #
# PQ                                                                    #
# ──────────────────────────────────────────────────────────────────── #

class PQ:
    def __init__(
        self, dim: int, M: int, K: int = 256, seed: int = 0,
        use_rht: bool = True,
    ) -> None:
        self.dim = dim
        self.pdim = padded_dim(dim) if use_rht else dim
        assert self.pdim % M == 0, f"pdim={self.pdim} not divisible by M={M}"
        self.M = M
        self.d_sub = self.pdim // M
        self.K = K
        self.seed = seed
        self.use_rht = use_rht
        self.codebooks = np.zeros((M, K, self.d_sub), dtype=np.float32)
        # uint8 if K ≤ 256, else uint16; we assume K ≤ 256 in this study.
        assert K <= 256
        self.codes = np.zeros((0, M), dtype=np.uint8)

    def _pre(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.use_rht:
            padded = np.zeros((len(X), self.pdim), dtype=np.float32)
            padded[:, : X.shape[1]] = X
            rot = rht(padded, self.seed)
            norms = np.linalg.norm(rot, axis=1, keepdims=True)
            return (rot / (norms + 1e-12)).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / (norms + 1e-12)).astype(np.float32)

    def fit(self, training: NDArray[np.float32]) -> None:
        rot = self._pre(training)
        for j in range(self.M):
            Xj = rot[:, j * self.d_sub : (j + 1) * self.d_sub]
            self.codebooks[j] = kmeans_mse(Xj, self.K, n_iters=15, seed=self.seed + j)

    def add(self, X: NDArray[np.float32]) -> None:
        rot = self._pre(X)
        codes = np.empty((len(X), self.M), dtype=np.uint8)
        for j in range(self.M):
            Xj = rot[:, j * self.d_sub : (j + 1) * self.d_sub]
            d2 = (
                (Xj ** 2).sum(1, keepdims=True)
                - 2 * Xj @ self.codebooks[j].T
                + (self.codebooks[j] ** 2).sum(1)[None, :]
            )
            codes[:, j] = d2.argmin(1).astype(np.uint8)
        self.codes = codes if len(self.codes) == 0 else np.vstack([self.codes, codes])

    def search(self, q: NDArray[np.float32], k: int) -> list[tuple[int, float]]:
        if self.use_rht:
            padded = np.zeros(self.pdim, dtype=np.float32)
            padded[: q.shape[-1]] = q
            q_rot = rht(padded[None, :], self.seed)[0]
        else:
            q_rot = q
        q_rot = q_rot / (np.linalg.norm(q_rot) + 1e-12)
        lut = np.empty((self.M, self.K), dtype=np.float32)
        for j in range(self.M):
            qj = q_rot[j * self.d_sub : (j + 1) * self.d_sub]
            lut[j] = self.codebooks[j] @ qj
        scores = np.zeros(len(self.codes), dtype=np.float32)
        for j in range(self.M):
            scores += lut[j][self.codes[:, j]]
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


# ──────────────────────────────────────────────────────────────────── #
# Metrics + harness                                                     #
# ──────────────────────────────────────────────────────────────────── #

def brute(q: NDArray[np.float32], c: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64], k: int) -> float:
    hits = sum(len(set(p[:k]) & set(t[:k].tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def eval_pq(
    *, dim: int, M: int, K: int, use_rht: bool,
    train: NDArray[np.float32], corpus: NDArray[np.float32],
    queries: NDArray[np.float32], truth: NDArray[np.int64],
    seeds: list[int], k: int,
) -> tuple[float, float, float]:
    rs = []
    total_t = 0.0
    for s in seeds:
        t0 = perf_counter()
        pq = PQ(dim=dim, M=M, K=K, seed=s, use_rht=use_rht)
        pq.fit(train)
        pq.add(corpus)
        pred = [[h[0] for h in pq.search(q, k=k)] for q in queries]
        rs.append(recall_at_k(pred, truth, k))
        total_t += perf_counter() - t0
    return float(np.mean(rs)), float(np.std(rs)), total_t / len(seeds)


# ──────────────────────────────────────────────────────────────────── #
# Main                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    cache = Path("experiments/.cache_scifact_bge_small.npy")
    if not cache.exists():
        print("run bench_multi_round_rht_real.py first to build the cache.")
        return
    vecs = np.load(cache).astype(np.float32)
    n_corpus, n_queries, k = 3000, 300, 10

    # Train-on-first-2000 / index-on-full-3000 split (prevents codebook
    # overfitting from skewing the comparison).
    train = vecs[:2000]
    corpus = vecs[:n_corpus]
    queries = vecs[n_corpus:n_corpus + n_queries]
    truth = brute(queries, corpus, k)
    d = corpus.shape[1]
    seeds = [0, 1, 2]

    # ── Snapvec baselines ──
    print(f"snapvec baselines (d={d}, pdim={padded_dim(d)}):")
    for b in [2, 3, 4]:
        idx = SnapIndex(dim=d, bits=b, normalized=True, seed=0)
        idx.add_batch(list(range(n_corpus)), corpus)
        pred = [[h[0] for h in idx.search(q, k=k)] for q in queries]
        bpv = padded_dim(d) * b // 8
        print(f"  b={b:<2}  ({bpv:>3d} B/vec tight):  recall@10 = {recall_at_k(pred, truth, k):.3f}")

    # ── PQ sweep over K ∈ {16, 64, 256} ──
    # Storage per vec (tight): M * log2(K) / 8 bytes.
    # Target a few matched storage points.
    targets = [
        # (K, log2K_bits)
        (16, 4),
        (64, 6),
        (256, 8),
    ]
    print("\nPQ (tight bytes/vec = M × log2(K) / 8)\n"
          "  with-RHT uses pdim=512 subspaces; no-RHT uses dim=384.\n"
          "  3 seeds, mean ± std.\n")

    for K, log2K in targets:
        print(f"=== K = {K}  ({log2K} bits/sub) ===")
        header = (f"  {'M':>4}  {'bpv':>4}  {'d/M (rht/raw)':>13}  "
                  f"{'rht recall':>13}  {'no-rht recall':>15}  "
                  f"{'t/seed rht':>10}  {'t/seed raw':>10}")
        print(header)

        # choose M values so pdim=512 AND dim=384 are each divisible (or skip)
        M_candidates = [16, 32, 48, 64, 96, 128, 192, 256, 384]
        for M in M_candidates:
            bpv = M * log2K // 8
            rht_ok = 512 % M == 0
            raw_ok = 384 % M == 0
            if not rht_ok and not raw_ok:
                continue
            cell_rht = cell_raw = "—"
            t_rht = t_raw = 0.0
            if rht_ok:
                m, s, t = eval_pq(
                    dim=d, M=M, K=K, use_rht=True,
                    train=train, corpus=corpus, queries=queries,
                    truth=truth, seeds=seeds, k=k,
                )
                cell_rht, t_rht = f"{m:.3f} ± {s:.3f}", t
            if raw_ok:
                m, s, t = eval_pq(
                    dim=d, M=M, K=K, use_rht=False,
                    train=train, corpus=corpus, queries=queries,
                    truth=truth, seeds=seeds, k=k,
                )
                cell_raw, t_raw = f"{m:.3f} ± {s:.3f}", t
            dsub = (f"{512//M if rht_ok else '—'} / "
                    f"{384//M if raw_ok else '—'}")
            print(f"  {M:>4}  {bpv:>4}  {dsub:>13}  {cell_rht:>13}  "
                  f"{cell_raw:>15}  {t_rht:>10.1f}s  {t_raw:>10.1f}s")
        print()


if __name__ == "__main__":
    main()

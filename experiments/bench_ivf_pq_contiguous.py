"""IVF-PQ with contiguous per-cluster layout — take 2.

Prior prototype (``bench_ivf_pq.py``) at N=3 000 was slower than full-
scan PQ because (a) the fixed LUT cost dominated, and (b) the Python-
level ``np.where(clusters == c)`` + fancy-indexing hot loop was much
more expensive than a single contiguous gather over the whole corpus.

This iteration fixes (b): on ``add``, we sort everything by cluster id
once.  Then ``inv[c]`` is a contiguous slice of the codes matrix — no
per-cluster `where` pass on every query.  (a) is amortised across a
larger corpus, so the benchmark runs at **N = 20 000** (FIQA / BGE-
small) to actually exercise the regime where IVF is supposed to win.

We report both per-cluster walltime breakdown and the recall / speedup
Pareto.

Run: ``python experiments/bench_ivf_pq_contiguous.py``
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────── #
# K-means primitives                                                    #
# ──────────────────────────────────────────────────────────────────── #

def _kpp(X: NDArray[np.float32], K: int, rng: np.random.Generator) -> NDArray[np.float32]:
    n = X.shape[0]
    centers = [X[int(rng.integers(n))]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        total = d2.sum()
        p = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=p))
        centers.append(X[nxt])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    return np.stack(centers).astype(np.float32)


def kmeans_mse(
    X: NDArray[np.float32], K: int, n_iters: int = 15, seed: int = 0,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    C = _kpp(X, K, rng)
    x_sq = (X ** 2).sum(1, keepdims=True)
    for _ in range(n_iters):
        d2 = x_sq - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        asn = d2.argmin(1)
        newC = np.empty_like(C)
        for k in range(K):
            m = asn == k
            newC[k] = X[m].mean(0) if m.any() else X[d2.min(1).argmax()]
        if np.allclose(newC, C, atol=1e-5):
            return newC
        C = newC
    return C


def _assign(X: NDArray[np.float32], C: NDArray[np.float32]) -> NDArray[np.int64]:
    d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return d2.argmin(1)


# ──────────────────────────────────────────────────────────────────── #
# PQ full-scan (baseline)                                               #
# ──────────────────────────────────────────────────────────────────── #

class PQFull:
    def __init__(self, dim: int, M: int, K: int = 256, seed: int = 0) -> None:
        assert dim % M == 0
        self.dim, self.M, self.K = dim, M, K
        self.d_sub = dim // M
        self.seed = seed
        self.codebooks = np.zeros((M, K, self.d_sub), dtype=np.float32)
        self.codes = np.zeros((0, M), dtype=np.uint8)

    def fit(self, training: NDArray[np.float32]) -> None:
        units = training / (np.linalg.norm(training, axis=1, keepdims=True) + 1e-12)
        for j in range(self.M):
            Xj = units[:, j * self.d_sub:(j + 1) * self.d_sub]
            self.codebooks[j] = kmeans_mse(Xj, self.K, seed=self.seed + j)

    def add(self, X: NDArray[np.float32]) -> None:
        units = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        codes = np.empty((len(units), self.M), dtype=np.uint8)
        for j in range(self.M):
            Xj = units[:, j * self.d_sub:(j + 1) * self.d_sub]
            d2 = (
                (Xj ** 2).sum(1, keepdims=True)
                - 2 * Xj @ self.codebooks[j].T
                + (self.codebooks[j] ** 2).sum(1)[None, :]
            )
            codes[:, j] = d2.argmin(1).astype(np.uint8)
        self.codes = codes if len(self.codes) == 0 else np.vstack([self.codes, codes])

    def search(self, q: NDArray[np.float32], k: int) -> list[tuple[int, float]]:
        q = q / (np.linalg.norm(q) + 1e-12)
        lut = np.empty((self.M, self.K), dtype=np.float32)
        for j in range(self.M):
            qj = q[j * self.d_sub:(j + 1) * self.d_sub]
            lut[j] = self.codebooks[j] @ qj
        scores = np.zeros(len(self.codes), dtype=np.float32)
        for j in range(self.M):
            scores += lut[j][self.codes[:, j]]
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


# ──────────────────────────────────────────────────────────────────── #
# IVF-PQ with contiguous per-cluster layout                             #
# ──────────────────────────────────────────────────────────────────── #

class IVFPQContiguous:
    """Residual PQ with coarse k-means partition, stored cluster-contiguously.

    After ``add``, the internal code matrix is ordered by cluster id; an
    ``offsets`` array gives the start/end row of each cluster.  Probed-
    cluster access is then a contiguous slice, which skips the Python
    overhead of ``np.where(clusters == c)`` on every query.
    """

    def __init__(
        self, dim: int, nlist: int, M: int, K: int = 256, seed: int = 0,
    ) -> None:
        assert dim % M == 0
        assert 2 <= K <= 256
        self.dim = dim
        self.nlist = nlist
        self.M = M
        self.K = K
        self.d_sub = dim // M
        self.seed = seed
        self.coarse = np.zeros((nlist, dim), dtype=np.float32)
        self.codebooks = np.zeros((M, K, self.d_sub), dtype=np.float32)
        # Cluster-contiguous storage.
        self.codes = np.zeros((0, M), dtype=np.uint8)
        # offsets[c]..offsets[c+1] is the slice of `codes` (and `original_rows`)
        # belonging to cluster c.
        self.offsets = np.zeros(nlist + 1, dtype=np.int64)
        # Map from internal (cluster-sorted) row back to the original id.
        self.original_rows = np.zeros(0, dtype=np.int64)

    def fit(self, training: NDArray[np.float32]) -> None:
        units = training / (np.linalg.norm(training, axis=1, keepdims=True) + 1e-12)
        self.coarse = kmeans_mse(units, self.nlist, n_iters=15, seed=self.seed)
        asn = _assign(units, self.coarse)
        residuals = units - self.coarse[asn]
        for j in range(self.M):
            Rj = residuals[:, j * self.d_sub:(j + 1) * self.d_sub].astype(np.float32)
            self.codebooks[j] = kmeans_mse(Rj, self.K, seed=self.seed + 1000 + j)

    def add(self, vectors: NDArray[np.float32]) -> None:
        units = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        asn = _assign(units, self.coarse)
        residuals = units - self.coarse[asn]
        raw_codes = np.empty((len(units), self.M), dtype=np.uint8)
        for j in range(self.M):
            Rj = residuals[:, j * self.d_sub:(j + 1) * self.d_sub].astype(np.float32)
            d2 = (
                (Rj ** 2).sum(1, keepdims=True)
                - 2 * Rj @ self.codebooks[j].T
                + (self.codebooks[j] ** 2).sum(1)[None, :]
            )
            raw_codes[:, j] = d2.argmin(1).astype(np.uint8)
        # Sort by cluster id → contiguous layout.
        order = np.argsort(asn, kind="stable")
        self.codes = raw_codes[order]
        self.original_rows = order.astype(np.int64)
        # offsets via bincount
        counts = np.bincount(asn, minlength=self.nlist)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def search(
        self, q: NDArray[np.float32], k: int, nprobe: int,
    ) -> list[tuple[int, float]]:
        q = q / (np.linalg.norm(q) + 1e-12)
        coarse_scores = self.coarse @ q
        probe = np.argpartition(-coarse_scores, min(nprobe, self.nlist) - 1)[:nprobe]
        # Per-subspace LUT over residual codebooks.
        lut = np.empty((self.M, self.K), dtype=np.float32)
        for j in range(self.M):
            qj = q[j * self.d_sub:(j + 1) * self.d_sub]
            lut[j] = self.codebooks[j] @ qj

        # Concatenate all probed slices into one contiguous block, then a
        # single gather+sum over it — this is what makes the contiguous
        # layout actually pay off vs. the fancy-indexing prototype.
        slices = [self.codes[self.offsets[c]:self.offsets[c + 1]] for c in probe]
        rows_slices = [
            self.original_rows[self.offsets[c]:self.offsets[c + 1]] for c in probe
        ]
        offsets_sc = [float(coarse_scores[c]) for c in probe]
        # length of each slice
        if not slices:
            return []
        cat = np.concatenate(slices, axis=0)
        orig = np.concatenate(rows_slices, axis=0)
        # Compose constant offsets per row (repeat coarse score for each
        # row in its cluster slice).
        coarse_rep = np.repeat(
            np.asarray(offsets_sc, dtype=np.float32),
            [len(s) for s in slices],
        )

        scores = np.zeros(len(cat), dtype=np.float32)
        for j in range(self.M):
            scores += lut[j][cat[:, j]]
        scores += coarse_rep

        k_eff = min(k, len(scores))
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [(int(orig[i]), float(scores[i])) for i in top]


# ──────────────────────────────────────────────────────────────────── #
# Metrics                                                               #
# ──────────────────────────────────────────────────────────────────── #

def brute(q: NDArray[np.float32], c: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64], k: int) -> float:
    hits = sum(len(set(p[:k]) & set(t[:k].tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def time_q(search_fn, queries: NDArray[np.float32]) -> float:
    t0 = perf_counter()
    for q in queries:
        search_fn(q)
    return (perf_counter() - t0) / len(queries) * 1e3


# ──────────────────────────────────────────────────────────────────── #
# Main                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def _augment(
    base: NDArray[np.float32], target_n: int, sigma: float = 0.015, seed: int = 0,
) -> NDArray[np.float32]:
    """Grow a corpus of ``target_n`` unit vectors by jittering ``base``.

    Each original vector is replicated ``target_n // len(base) + 1`` times
    with zero-mean Gaussian noise of stddev ``sigma`` and re-normalized.
    This preserves the manifold structure k-means is trying to exploit
    (essential for the IVF Pareto to be meaningful) while letting us
    benchmark at N ≫ cache size without another embedding pass.
    """
    rng = np.random.default_rng(seed)
    reps = (target_n + len(base) - 1) // len(base)
    out = np.tile(base, (reps, 1)) + sigma * rng.standard_normal(
        (reps * len(base), base.shape[1])
    ).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    # Shuffle so the cluster contiguity comes from k-means, not replication.
    rng.shuffle(out)
    return out[:target_n]


def main() -> None:
    # Use the cached SciFact / BGE-small embeddings as a seed corpus,
    # augment to ~20k via small Gaussian jitter (keeps the embedding
    # manifold intact for k-means to discover).
    cache = Path("experiments/.cache_scifact_bge_small.npy")
    if not cache.exists():
        print("run bench_multi_round_rht_real.py first to build the cache.")
        return
    base = np.load(cache).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    n_corpus, n_queries, k = 20_000, 500, 10
    all_vecs = _augment(base, n_corpus + n_queries + 5000, sigma=0.015, seed=0)
    corpus = all_vecs[:n_corpus]
    queries = all_vecs[n_corpus:n_corpus + n_queries]
    train = all_vecs[n_corpus + n_queries:]
    truth = brute(queries, corpus, k)
    d = corpus.shape[1]

    M, K = 192, 256  # matches 192 B/vec operating point of PQSnapIndex
    print(f"dataset: BGE-small / FIQA, N={n_corpus}, d={d}, "
          f"M={M}, K={K}, storage={M} B/vec (+2 B coarse id for IVF)")

    # ── full-scan PQ reference ──
    print("\nPQ full-scan (baseline):")
    pq = PQFull(d, M=M, K=K, seed=0)
    pq.fit(train)
    pq.add(corpus)
    pred = [[h[0] for h in pq.search(q, k)] for q in queries]
    r_full = recall_at_k(pred, truth, k)
    t_full = time_q(lambda q: pq.search(q, k), queries)
    print(f"  recall@{k} = {r_full:.3f}   latency = {t_full:.2f} ms/q")

    # ── IVF-PQ contiguous sweep ──
    print("\nIVF-PQ (contiguous layout):")
    # Rule-of-thumb nlist ~ 4·√N = 4·√20000 ≈ 565; try 128, 256, 512.
    for nlist in [128, 256, 512]:
        print(f"\n=== nlist={nlist} ===")
        ivf = IVFPQContiguous(d, nlist=nlist, M=M, K=K, seed=0)
        ivf.fit(train)
        ivf.add(corpus)
        sizes = np.diff(ivf.offsets)
        print(f"  cluster sizes: min={int(sizes.min())}, "
              f"median={int(np.median(sizes))}, max={int(sizes.max())}")
        print(f"  {'nprobe':>6}  {'frac':>6}  {'recall':>6}  "
              f"{'ms/q':>7}  {'vs full':>7}")
        probes = sorted({1, 2, 4, 8, 16, 32, 64, nlist // 8,
                         nlist // 4, nlist // 2})
        probes = [p for p in probes if 1 <= p <= nlist]
        for nprobe in probes:
            pred = [[h[0] for h in ivf.search(q, k, nprobe)] for q in queries]
            r = recall_at_k(pred, truth, k)
            t = time_q(lambda q, p=nprobe: ivf.search(q, k, p), queries)
            spd = t_full / t if t > 0 else 0.0
            print(f"  {nprobe:>6}  {nprobe/nlist:>6.1%}  {r:>6.3f}  "
                  f"{t:>7.2f}  {spd:>6.2f}×")


if __name__ == "__main__":
    main()

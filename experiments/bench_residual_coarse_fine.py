"""Residual coarse-to-fine TurboQuant.

Idea: quantize x with b₁ bits → x̃₁.  Quantize the residual r = x − x̃₁
with b₂ bits (after rescaling by the residual std, since Lloyd-Max is
optimal on unit-variance Gaussians).  Storage = b₁ + b₂ bits/coord,
but using codebooks snapvec already ships ({2,3,4}).

Two angles to evaluate:

1. **Full reconstruction:**  decode x̃ ≈ x̃₁ + σ_r · r̃ and search.
   Tests whether cascading two Lloyd-Max stages gives better
   distortion than a single high-rate stage at equal bits.  Theory
   says no for iid Gaussian sources (single-stage is rate-optimal)
   but the residual is non-Gaussian, so empirics matter.

2. **Coarse + rerank:**  search at b₁ bits over the whole corpus to
   get a candidate set of size M, then rerank those M with the full
   b₁+b₂ representation.  This is the genuinely useful operating
   mode — fast coarse pass + accurate rerank — independent of the
   distortion theory.

Run: ``python experiments/bench_residual_coarse_fine.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from snapvec import SnapIndex
from snapvec._codebooks import get_codebook
from snapvec._rotation import padded_dim, rht


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                               #
# ──────────────────────────────────────────────────────────────────── #

def _quantize(
    x: NDArray[np.float32], bits: int
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize ``x`` coord-wise with Lloyd-Max @ ``bits``.

    Returns (codes uint8, decoded float32) for the same array of values.
    Assumes ``x`` is approximately N(0, 1) distributed.
    """
    cent, thr = get_codebook(bits)
    idx = np.clip(np.searchsorted(thr, x), 0, (2 ** bits) - 1).astype(np.uint8)
    return idx, cent[idx]


def _rotate(x: NDArray[np.float32], pdim: int, seed: int) -> NDArray[np.float32]:
    padded = np.zeros((len(x), pdim), dtype=np.float32)
    padded[:, :x.shape[1]] = x
    return rht(padded, seed)


# ──────────────────────────────────────────────────────────────────── #
# Residual index                                                        #
# ──────────────────────────────────────────────────────────────────── #

class ResidualIndex:
    """Two-stage Lloyd-Max: coarse @ b1 bits, residual @ b2 bits."""

    def __init__(self, dim: int, b1: int, b2: int, seed: int = 0) -> None:
        self.dim = dim
        self.pdim = padded_dim(dim)
        self.b1, self.b2 = b1, b2
        self.seed = seed
        self.codes1: NDArray[np.uint8] = np.zeros((0, self.pdim), dtype=np.uint8)
        self.codes2: NDArray[np.uint8] = np.zeros((0, self.pdim), dtype=np.uint8)
        self.norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # Residual std (scalar, estimated from first batch) — needed to
        # rescale the residual into ~N(0,1) before the 2nd Lloyd-Max stage.
        self.sigma_r: float = 1.0

    def add(self, vectors: NDArray[np.float32]) -> None:
        rot = _rotate(vectors, self.pdim, self.seed)
        norms = np.linalg.norm(rot, axis=1)
        # Normalize so each vector has ~unit per-coord variance on average.
        # (Coords after RHT share the same energy budget.)
        rot_unit = rot / (norms[:, None] + 1e-12) * np.sqrt(self.pdim)

        c1, dec1 = _quantize(rot_unit, self.b1)
        residual = rot_unit - dec1
        if len(self.norms) == 0:  # first call: estimate residual std
            self.sigma_r = float(residual.std())
        # Rescale residual to ~unit variance before 2nd-stage Lloyd-Max.
        c2, _ = _quantize(residual / (self.sigma_r + 1e-12), self.b2)

        self.codes1 = np.vstack([self.codes1, c1])
        self.codes2 = np.vstack([self.codes2, c2])
        self.norms = np.concatenate([self.norms, norms])

    def _decode_coarse(self) -> NDArray[np.float32]:
        cent, _ = get_codebook(self.b1)
        return cent[self.codes1]

    def _decode_full(self, rows: NDArray[np.int64] | slice) -> NDArray[np.float32]:
        cent1, _ = get_codebook(self.b1)
        cent2, _ = get_codebook(self.b2)
        return cent1[self.codes1[rows]] + self.sigma_r * cent2[self.codes2[rows]]

    def search_full(
        self, query: NDArray[np.float32], k: int
    ) -> list[tuple[int, float]]:
        q_rot = _rotate(query[None, :], self.pdim, self.seed)[0]
        q_unit = q_rot / (np.linalg.norm(q_rot) + 1e-12)
        decoded = self._decode_full(slice(None)) / np.sqrt(self.pdim)
        sims = decoded @ q_unit
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def search_coarse_rerank(
        self, query: NDArray[np.float32], k: int, M: int
    ) -> list[tuple[int, float]]:
        """Coarse-pass top-M (b1 bits) then full-reconstruction rerank."""
        q_rot = _rotate(query[None, :], self.pdim, self.seed)[0]
        q_unit = q_rot / (np.linalg.norm(q_rot) + 1e-12)
        coarse = self._decode_coarse() / np.sqrt(self.pdim)
        cscores = coarse @ q_unit
        M = min(M, len(cscores))
        cand = np.argpartition(-cscores, M - 1)[:M]
        fine = self._decode_full(cand) / np.sqrt(self.pdim)
        fscores = fine @ q_unit
        order = np.argsort(-fscores)[:k]
        return [(int(cand[i]), float(fscores[i])) for i in order]


# ──────────────────────────────────────────────────────────────────── #
# Metrics                                                               #
# ──────────────────────────────────────────────────────────────────── #

def brute(q: NDArray[np.float32], c: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64], k: int) -> float:
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k]) & set(t[:k].tolist()))
    return hits / (len(pred) * k)


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
    corpus, queries = vecs[:n_corpus], vecs[n_corpus:n_corpus + n_queries]
    truth = brute(queries, corpus, k)
    d = corpus.shape[1]
    ids = list(range(n_corpus))

    # ── Baseline: uniform single-stage SnapIndex ──
    print("Uniform baseline (SnapIndex):")
    base: dict[int, float] = {}
    for b in [2, 3, 4]:
        idx = SnapIndex(dim=d, bits=b, normalized=True, seed=0)
        idx.add_batch(ids, corpus)
        pred = [[h[0] for h in idx.search(q, k=k)] for q in queries]
        base[b] = recall_at_k(pred, truth, k)
        print(f"  b={b}: recall@10 = {base[b]:.3f}   (bits/coord = {b})")

    # ── Residual full reconstruction ──
    print("\nResidual full-reconstruction recall@10:")
    print(f"  {'b1':>3} {'b2':>3}  {'total':>5}   {'recall':>6}   "
          f"vs uniform (same or nearest bits)")
    rimap: dict[tuple[int, int], ResidualIndex] = {}
    for b1, b2 in [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3)]:
        ridx = ResidualIndex(dim=d, b1=b1, b2=b2, seed=0)
        ridx.add(corpus)
        rimap[(b1, b2)] = ridx
        pred = [[h[0] for h in ridx.search_full(q, k=k)] for q in queries]
        r = recall_at_k(pred, truth, k)
        tot = b1 + b2
        ref = base.get(min(tot, 4), base[4])
        print(f"  {b1:>3} {b2:>3}   {tot:>2}      {r:.3f}      "
              f"(uniform b≤4 best = {max(base.values()):.3f}; sigma_r="
              f"{ridx.sigma_r:.3f})")

    # ── Coarse + rerank ──
    print("\nCoarse-pass (b1) + top-M rerank with residual (b1+b2):")
    print(f"  {'b1':>3} {'b2':>3}  {'M':>4}   {'recall':>6}")
    for (b1, b2) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        ridx = rimap.get((b1, b2)) or ResidualIndex(dim=d, b1=b1, b2=b2, seed=0)
        if (b1, b2) not in rimap:
            ridx.add(corpus)
        for M in [50, 100, 200, 500]:
            pred = [[h[0] for h in ridx.search_coarse_rerank(q, k=k, M=M)]
                    for q in queries]
            r = recall_at_k(pred, truth, k)
            print(f"  {b1:>3} {b2:>3}  {M:>4}   {r:.3f}")


if __name__ == "__main__":
    main()

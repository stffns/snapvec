"""Measure fit() and search_batch() latency impact of the astype cleanup.

Only worth shipping if the delta is above the +/- 1% noise floor we
saw on the Bolt einsum PR.  Uses a realistic FIQA-scale synthetic
corpus (N=50k, dim=384) so the subspace k-means dominates and the
astype-per-subspace path in fit is actually hot.
"""
from __future__ import annotations

import time

import numpy as np

from snapvec import IVFPQSnapIndex, PQSnapIndex


def bench_fit(dim: int = 384, n: int = 50_000, M: int = 96) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    # Warm up to amortise first-call overhead
    warm = PQSnapIndex(dim=dim, M=M, seed=0, normalized=True)
    warm.fit(vecs[:5000])

    times = []
    for _ in range(3):
        idx = IVFPQSnapIndex(dim=dim, M=M, nlist=64, seed=0, normalized=True)
        t0 = time.perf_counter()
        idx.fit(vecs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), float(np.mean(times))


def bench_search_batch(
    dim: int = 384, n: int = 50_000, M: int = 96, B: int = 100,
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    queries = rng.standard_normal((B, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    idx = IVFPQSnapIndex(dim=dim, M=M, nlist=64, seed=0, normalized=True)
    idx.fit(vecs[:20_000])
    idx.add_batch([str(i) for i in range(n)], vecs)

    # Warm up
    for _ in range(3):
        _ = idx.search_batch(queries, k=10, nprobe=8)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = idx.search_batch(queries, k=10, nprobe=8)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), float(np.mean(times))


if __name__ == "__main__":
    print("--- fit() benchmark, N=50k M=96 ---")
    fit_min, fit_mean = bench_fit()
    print(f"best: {fit_min*1000:7.1f} ms  mean: {fit_mean*1000:7.1f} ms")

    print("--- search_batch(B=100) N=50k M=96 nprobe=8 ---")
    sb_min, sb_mean = bench_search_batch()
    print(f"best: {sb_min*1000:7.2f} ms  mean: {sb_mean*1000:7.2f} ms")

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


def bench_fit(
    dim: int = 384, n: int = 50_000, M: int = 96, use_rht: bool = False,
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    # Warm up against the same IVFPQ code path we are about to time
    # (previous version warmed PQSnapIndex, which has no coarse stage
    # and a different k-means call pattern -- BLAS caches ended up
    # warming on the first timed iteration instead of the warmup).
    warm = IVFPQSnapIndex(
        dim=dim, M=M, nlist=16, seed=0, normalized=True, use_rht=use_rht,
    )
    warm.fit(vecs[:5000])

    times = []
    for _ in range(3):
        idx = IVFPQSnapIndex(
            dim=dim, M=M, nlist=64, seed=0, normalized=True, use_rht=use_rht,
        )
        t0 = time.perf_counter()
        idx.fit(vecs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), float(np.mean(times))


def bench_search_batch(
    dim: int = 384, n: int = 50_000, M: int = 96, B: int = 100,
    use_rht: bool = False,
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    queries = rng.standard_normal((B, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    idx = IVFPQSnapIndex(
        dim=dim, M=M, nlist=64, seed=0, normalized=True, use_rht=use_rht,
    )
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


def bench_snap_add_batch(
    dim: int = 384, n: int = 10_000, bits: int = 4, use_prod: bool = False,
) -> tuple[float, float]:
    """Covers the ``scaled`` / ``residual_norms`` hot paths in
    ``SnapIndex.add_batch`` that the IVFPQ benches do not exercise.
    """
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    warm = SnapIndex(dim=dim, bits=bits, seed=0, use_prod=use_prod)
    warm.add_batch(list(range(1000)), vecs[:1000])

    times = []
    for _ in range(3):
        idx = SnapIndex(dim=dim, bits=bits, seed=0, use_prod=use_prod)
        t0 = time.perf_counter()
        idx.add_batch(list(range(n)), vecs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), float(np.mean(times))


if __name__ == "__main__":
    print("--- IVFPQ fit() benchmark, N=50k M=96 ---")
    fit_min, fit_mean = bench_fit()
    print(f"  no RHT      best: {fit_min*1000:7.1f} ms  mean: {fit_mean*1000:7.1f} ms")
    fit_min, fit_mean = bench_fit(use_rht=True)
    print(f"  use_rht     best: {fit_min*1000:7.1f} ms  mean: {fit_mean*1000:7.1f} ms")

    print("--- IVFPQ search_batch(B=100) N=50k M=96 nprobe=8 ---")
    sb_min, sb_mean = bench_search_batch()
    print(f"  no RHT      best: {sb_min*1000:7.2f} ms  mean: {sb_mean*1000:7.2f} ms")
    sb_min, sb_mean = bench_search_batch(use_rht=True)
    print(f"  use_rht     best: {sb_min*1000:7.2f} ms  mean: {sb_mean*1000:7.2f} ms")

    print("--- SnapIndex.add_batch N=10k ---")
    ab_min, ab_mean = bench_snap_add_batch(use_prod=False)
    print(f"  use_prod=F  best: {ab_min*1000:7.1f} ms  mean: {ab_mean*1000:7.1f} ms")
    ab_min, ab_mean = bench_snap_add_batch(use_prod=True)
    print(f"  use_prod=T  best: {ab_min*1000:7.1f} ms  mean: {ab_mean*1000:7.1f} ms")

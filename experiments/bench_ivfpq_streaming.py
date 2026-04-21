"""Streaming-ingest profile for IVFPQSnapIndex.add_batch.

Measures the cost of many small add_batch calls growing the index
from 0 to ~N.  The theoretical concern is that add_batch re-sorts
the whole corpus by cluster id every call; at constant batch size
this is O(N log N) per call, O(N^2 log N) amortised.

Run: python experiments/bench_ivfpq_streaming.py
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from snapvec import IVFPQSnapIndex


DIM = 128
M = 16
K_CENT = 64
NLIST = 32
SEED = 0


def clustered(n: int, dim: int, *, seed: int, n_clusters: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3
    assign = rng.integers(0, n_clusters, size=n)
    jitter = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    return centers[assign] + jitter


def main() -> None:
    total_N = 50_000
    batch_size = 500

    corpus = clustered(total_N, DIM, seed=SEED)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12

    idx = IVFPQSnapIndex(
        dim=DIM, nlist=NLIST, M=M, K=K_CENT, normalized=True, seed=SEED,
    )
    idx.fit(corpus[:5000])  # train once

    print(f"Streaming {total_N} into IVFPQ, batch_size={batch_size}")
    print(f"{'current N':>10}  {'batch n':>7}  {'add_batch ms':>12}  {'ms / row':>10}")
    print("-" * 48)

    n_done = 0
    timings = []
    while n_done < total_N:
        end = min(n_done + batch_size, total_N)
        batch_vecs = corpus[n_done:end]
        batch_ids = list(range(n_done, end))
        t0 = perf_counter()
        idx.add_batch(batch_ids, batch_vecs)
        dt_ms = (perf_counter() - t0) * 1e3
        timings.append(dt_ms)
        if n_done % 5000 == 0 or end >= total_N:
            per_row_us = dt_ms / (end - n_done) * 1e3
            print(f"{end:>10}  {end - n_done:>7}  {dt_ms:>12.1f}  {per_row_us:>10.2f} us")
        n_done = end

    total_ms = sum(timings)
    print()
    print(f"total: {total_ms:.0f} ms over {len(timings)} batches")
    # Quadratic cost check: mean of last 10 batches / mean of first 10 batches
    first = np.mean(timings[:10])
    last = np.mean(timings[-10:])
    ratio = last / first if first > 0 else float('inf')
    print(f"last-10 / first-10 ratio: {ratio:.1f}x  "
          f"(ideal O(1) = 1x, O(N) batches -> linear growth)")


if __name__ == "__main__":
    main()

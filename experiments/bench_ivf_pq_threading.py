"""A/B for ``num_threads`` on IVFPQSnapIndex.search_batch.

Reuses the FIQA recall harness; reports each nprobe at num_threads
in {1, 2, 4, 8} so the speedup curve and the small-nprobe overhead
crossover can be read off a single table.

Threading lives on ``search_batch`` (not ``search``) because batch-level
fan-out amortises Python overhead over a whole query's worth of work.
The single-query ``search()`` API is serial by design; that is where
latency per call is minimised against NumPy's BLAS pool.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from snapvec import IVFPQSnapIndex


CORPUS_CANDIDATES = [
    Path("experiments/.cache_fiqa_corpus_bge_small.npy"),
    Path("experiments/.cache_fiqa_bge_small.npy"),
]
QUERIES_PATH = Path("experiments/.cache_fiqa_queries_bge_small.npy")

NLIST = 512
M = 192
K = 256
KK = 10
NPROBES = [4, 8, 16, 32, 64, 128, 256]
THREAD_COUNTS = [1, 2, 4, 8]
BATCH_SIZE = 128
SEED = 0


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def time_batch(
    idx: IVFPQSnapIndex,
    queries: NDArray[np.float32],
    nprobe: int,
    num_threads: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, float]:
    """Return (ms_per_query, total_elapsed_s) averaged over all batches."""
    # Warm-up: first batch warms LUT + gather caches.  The thread-pool
    # executor itself is only created when num_threads > 1.
    _ = idx.search_batch(
        queries[:batch_size], k=KK, nprobe=nprobe, num_threads=num_threads,
    )

    n = len(queries)
    t0 = perf_counter()
    for start in range(0, n, batch_size):
        chunk = queries[start : start + batch_size]
        idx.search_batch(chunk, k=KK, nprobe=nprobe, num_threads=num_threads)
    elapsed = perf_counter() - t0
    ms_per_q = elapsed / n * 1e3
    return ms_per_q, elapsed


def main() -> None:
    corpus = np.load(find_corpus()).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:512]  # enough batches to amortise scheduling

    print(f"corpus  : {corpus.shape}")
    print(f"queries : {queries.shape}  (batch_size={BATCH_SIZE})")

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    print(f"\nfit({len(corpus)}) + add_batch...")
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s")

    header = "nprobe" + "".join(f"  t={t} (ms/q)" for t in THREAD_COUNTS) + "  best-speedup"
    print()
    print(header)
    print("-" * len(header))
    for nprobe in NPROBES:
        row = []
        for t in THREAD_COUNTS:
            # The executor is lazily created and locked to the first
            # num_threads value.  Call close() between t settings to
            # release the executor so it can be recreated for the next t.
            idx.close()
            ms, _ = time_batch(idx, queries, nprobe, t)
            row.append(ms)
        base = row[0]
        best = min(row)
        speedup = base / best if best > 0 else 0.0
        cells = "".join(f"      {ms:5.2f}" for ms in row)
        print(f"{nprobe:>5}{cells}       {speedup:4.2f}x")

    idx.close()


if __name__ == "__main__":
    main()

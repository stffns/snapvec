"""A/B for search_batch on the FIQA harness.

Per-query loop vs batched API at batch sizes ∈ {1, 8, 32, 128, 500},
each with num_threads ∈ {1, 4}.  Reports throughput (queries / sec)
and effective ms / query.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

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
NPROBE = 32     # ~0.89 recall point on this corpus
BATCHES = [1, 8, 32, 128, 500]
SEED = 0


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def time_loop(idx, queries):
    for q in queries[:5]:
        idx.search(q, k=KK, nprobe=NPROBE)
    t0 = perf_counter()
    for q in queries:
        idx.search(q, k=KK, nprobe=NPROBE)
    return perf_counter() - t0


def time_batch(idx, queries, num_threads):
    # Warm
    idx.search_batch(queries[:8], k=KK, nprobe=NPROBE, num_threads=num_threads)
    t0 = perf_counter()
    idx.search_batch(queries, k=KK, nprobe=NPROBE, num_threads=num_threads)
    return perf_counter() - t0


def main() -> None:
    corpus = np.load(find_corpus()).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    print(f"corpus  : {corpus.shape}")
    print(f"queries : {queries.shape}")

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    print("\nfit + add…")
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s\n")

    print(f"@ nprobe={NPROBE}  (≈ 0.89 recall on this corpus)\n")
    print(f"{'batch':>5}  {'loop ms/q':>10}  {'batch t=1 ms/q':>15}  "
          f"{'speedup':>8}  {'batch t=4 ms/q':>15}  {'speedup':>8}")
    print("-" * 80)
    for B in BATCHES:
        if B > len(queries):
            continue
        sub = queries[:B]
        t_loop = time_loop(idx, sub)
        t_b1 = time_batch(idx, sub, num_threads=1)
        t_b4 = time_batch(idx, sub, num_threads=4)
        loop_ms = t_loop * 1e3 / B
        b1_ms = t_b1 * 1e3 / B
        b4_ms = t_b4 * 1e3 / B
        sp1 = loop_ms / b1_ms if b1_ms > 0 else 0.0
        sp4 = loop_ms / b4_ms if b4_ms > 0 else 0.0
        print(f"{B:>5}  {loop_ms:>10.3f}  {b1_ms:>15.3f}  "
              f"{sp1:>7.2f}×  {b4_ms:>15.3f}  {sp4:>7.2f}×")


if __name__ == "__main__":
    main()

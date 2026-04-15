"""Proper recall@10 vs nprobe sweep on real BGE-small / FIQA.

Why this exists: the previous N=1M benches used augmented copies of
corpus vectors as queries.  That made every query's true top-10 a
cluster of ~19 jitter copies all assigned to the same coarse cluster,
so nprobe=16 already covered the full candidate set and increasing
``nprobe`` did not move recall (we saw it pinned at 0.758 with
SciFact seeds and 0.869 with FIQA seeds).

This bench uses the *real* BeIR / FIQA queries split — independent
questions that were never part of the corpus — so the IVF probe
behaviour is exercised correctly: a query lands in cluster A, but its
true top-10 may be spread across clusters {A, B, C, …} depending on
how the corpus geometry partitions.  Recall is then a real function
of nprobe.

Run:

    1. Run experiments/_colab_embed_corpus.py with SPLITS=["corpus",
       "queries"] on Colab Pro / A100; download both .npy files.
    2. Place them at:
         experiments/.cache_fiqa_corpus_bge_small.npy
         experiments/.cache_fiqa_queries_bge_small.npy
       (legacy name experiments/.cache_fiqa_bge_small.npy also accepted
       for the corpus, since that's what the original colab script
       produced.)
    3. python experiments/bench_ivf_pq_fiqa_recall.py
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from snapvec import IVFPQSnapIndex


CORPUS_CANDIDATES = [
    Path("experiments/.cache_fiqa_corpus_bge_small.npy"),
    Path("experiments/.cache_fiqa_bge_small.npy"),  # legacy
]
QUERIES_PATH = Path("experiments/.cache_fiqa_queries_bge_small.npy")

NLIST = 512        # ≈ 2·√N at N=57k; avg cluster size ~110
M = 192
K = 256
KK = 10
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SEED = 0


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "missing FIQA corpus cache.  Run _colab_embed_corpus.py "
        "(SPLITS includes 'corpus') and place the .npy in experiments/."
    )


def brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32],
) -> NDArray[np.int64]:
    """Exact top-KK across the whole corpus.  At N=57k × Q=6.6k this
    is a single ~1.5 GFLOP matmul — sub-second."""
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub_scores = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64]) -> float:
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def time_queries(idx: IVFPQSnapIndex, queries: NDArray[np.float32], nprobe: int):
    for q in queries[:5]:
        idx.search(q, k=KK, nprobe=nprobe)
    times: list[float] = []
    pred: list[list[int]] = []
    for q in queries:
        t0 = perf_counter()
        hits = idx.search(q, k=KK, nprobe=nprobe)
        times.append(perf_counter() - t0)
        pred.append([h[0] for h in hits])
    times.sort()
    return pred, float(np.mean(times) * 1e3), float(times[int(0.95 * len(times))] * 1e3)


def main() -> None:
    corpus_path = find_corpus()
    if not QUERIES_PATH.exists():
        raise SystemExit(
            f"missing FIQA queries cache at {QUERIES_PATH}.  Re-run "
            "_colab_embed_corpus.py with SPLITS=['corpus', 'queries']."
        )
    corpus = np.load(corpus_path).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    print(f"corpus  : {corpus.shape}  from {corpus_path}", flush=True)
    print(f"queries : {queries.shape}  from {QUERIES_PATH}", flush=True)

    # FIQA has ~6.6k queries; 200 is plenty for a stable mean+p95
    # without making cProfile + brute-force slow.
    n_q = min(500, len(queries))
    queries = queries[:n_q]

    print(f"\nbrute-force ground truth ({n_q} × {len(corpus)})…", flush=True)
    t0 = perf_counter()
    truth = brute_topk(queries, corpus)
    print(f"  done in {perf_counter() - t0:.2f}s", flush=True)

    print(f"\n=== IVFPQSnapIndex N={len(corpus)} d={corpus.shape[1]} "
          f"nlist={NLIST} M={M} K={K} ===", flush=True)
    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    print(f"\nfit({len(corpus)}) on the full corpus…", flush=True)
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"add_batch({len(corpus)})…", flush=True)
    t0 = perf_counter()
    idx.add_batch(list(range(len(corpus))), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    sizes = np.diff(idx._offsets)
    print(f"\ncluster sizes: min={int(sizes.min())} median={int(np.median(sizes))} "
          f"max={int(sizes.max())} empty={int((sizes == 0).sum())}", flush=True)

    print(f"\n{'nprobe':>6}  {'visited':>8}  {'recall@10':>10}  "
          f"{'mean ms':>9}  {'p95 ms':>9}", flush=True)
    print("-" * 55, flush=True)
    for nprobe in NPROBES:
        if nprobe > NLIST:
            continue
        pred, mean_ms, p95_ms = time_queries(idx, queries, nprobe)
        r = recall_at_k(pred, truth)
        visited = int(sizes.mean() * nprobe)
        print(f"{nprobe:>6}  {visited:>8}  {r:>10.3f}  "
              f"{mean_ms:>9.2f}  {p95_ms:>9.2f}", flush=True)


if __name__ == "__main__":
    main()

"""v0.9.0 definitive benchmark on real FIQA/BGE-small embeddings.

Measures recall@10, latency, and rerank performance using real
embeddings (not random vectors) for publishable numbers.

Requires:
  experiments/.cache_fiqa_bge_small.npy        (corpus, ~57K x 384)
  experiments/.cache_fiqa_queries_bge_small.npy (queries, ~6.6K x 384)
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from snapvec import PQSnapIndex, IVFPQSnapIndex

CORPUS_CANDIDATES = [
    Path("experiments/.cache_fiqa_corpus_bge_small.npy"),
    Path("experiments/.cache_fiqa_bge_small.npy"),
]
QUERIES_PATH = Path("experiments/.cache_fiqa_queries_bge_small.npy")

M = 192
K = 256
KK = 10
NLIST = 512
NPROBES = [4, 8, 16, 32, 64, 128, 256]
N_QUERIES = 500
SEED = 0


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("Missing FIQA corpus .npy -- see docstring.")


def brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32],
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub_scores = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64]) -> float:
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def main() -> None:
    corpus_path = find_corpus()
    if not QUERIES_PATH.exists():
        raise SystemExit(f"Missing FIQA queries .npy at {QUERIES_PATH}")

    corpus = np.load(corpus_path).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:N_QUERIES]

    N = len(corpus)
    dim = corpus.shape[1]
    print(f"snapvec v0.9.0 -- FIQA/BGE-small Benchmark")
    print(f"corpus: {N:,} x {dim}  |  queries: {len(queries)}")
    print(f"M={M}, K={K}, nlist={NLIST}")

    # Ground truth
    print(f"\nBrute-force ground truth ({len(queries)} x {N:,})...", flush=True)
    t0 = perf_counter()
    truth = brute_topk(queries, corpus)
    print(f"  done in {perf_counter() - t0:.2f}s")

    # ── PQSnapIndex ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PQSnapIndex  (N={N:,}, dim={dim}, M={M}, K={K})")
    print(f"{'='*70}")

    pq = PQSnapIndex(dim=dim, M=M, K=K, normalized=True, seed=SEED)
    t0 = perf_counter()
    pq.fit(corpus)
    fit_s = perf_counter() - t0

    t0 = perf_counter()
    pq.add_batch(list(range(N)), corpus)
    add_s = perf_counter() - t0

    stats = pq.stats()
    print(f"  fit:  {fit_s:.1f}s  |  add: {add_s:.1f}s  |  "
          f"{stats['bytes_per_vec']} B/vec")

    # Warm up + measure
    for q in queries[:10]:
        pq.search(q, k=KK)
    pq_pred = []
    t0 = perf_counter()
    for q in queries:
        hits = pq.search(q, k=KK)
        pq_pred.append([h[0] for h in hits])
    pq_us = (perf_counter() - t0) / len(queries) * 1e6
    pq_rec = recall_at_k(pq_pred, truth)
    print(f"\n  recall@10: {pq_rec:.3f}  |  {pq_us:.0f} us/query")

    # ── IVFPQSnapIndex ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"IVFPQSnapIndex  (N={N:,}, nlist={NLIST}, M={M}, K={K})")
    print(f"{'='*70}")

    ivf = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=M, K=K,
        normalized=True, keep_full_precision=True, seed=SEED,
    )
    t0 = perf_counter()
    ivf.fit(corpus, kmeans_iters=15)
    fit_s = perf_counter() - t0

    t0 = perf_counter()
    ivf.add_batch(list(range(N)), corpus)
    add_s = perf_counter() - t0

    stats = ivf.stats()
    sizes = np.diff(ivf._offsets)
    print(f"  fit:  {fit_s:.1f}s  |  add: {add_s:.1f}s  |  "
          f"{stats['bytes_per_vec']} B/vec  "
          f"({stats.get('bytes_per_vec_codes_only', '?')} codes-only)")
    print(f"  clusters: min={int(sizes.min())} median={int(np.median(sizes))} "
          f"max={int(sizes.max())} empty={int((sizes == 0).sum())}")

    # Warm up
    for q in queries[:10]:
        ivf.search(q, k=KK, nprobe=32)

    print(f"\n  {'nprobe':>6}  {'recall@10':>10}  {'us/query':>10}"
          f"  {'rerank rec':>10}  {'rerank us':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for nprobe in NPROBES:
        if nprobe > NLIST:
            continue

        # PQ-only
        pred_pq = []
        t0 = perf_counter()
        for q in queries:
            hits = ivf.search(q, k=KK, nprobe=nprobe)
            pred_pq.append([h[0] for h in hits])
        us_pq = (perf_counter() - t0) / len(queries) * 1e6
        rec_pq = recall_at_k(pred_pq, truth)

        # With rerank
        pred_rr = []
        t0 = perf_counter()
        for q in queries:
            hits = ivf.search(q, k=KK, nprobe=nprobe, rerank_candidates=100)
            pred_rr.append([h[0] for h in hits])
        us_rr = (perf_counter() - t0) / len(queries) * 1e6
        rec_rr = recall_at_k(pred_rr, truth)

        print(f"  {nprobe:>6}  {rec_pq:>10.3f}  {us_pq:>10.0f}"
              f"  {rec_rr:>10.3f}  {us_rr:>10.0f}")


if __name__ == "__main__":
    main()

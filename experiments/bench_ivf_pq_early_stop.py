"""A/B for early_stop on the FIQA recall sweep.

Same setup as bench_ivf_pq_fiqa_recall.py but reports each nprobe
twice -- once with early_stop=False, once with early_stop=True -- so
the latency win and recall delta of the short-circuit can be read
directly off a single table.

NEGATIVE RESULT (2026-04-21): early_stop is slower than the batched
full-scan at every nprobe on FIQA (0.32x-0.98x).  Recall matches
(the bound is strict) but the stop condition rarely fires because
FIQA's ground-truth top-k spans many clusters.  See ROADMAP.md
`Parked / explored` for the detailed findings.  Code is kept on the
`feat/ivfpq-early-stop` branch.
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
SEED = 0


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def brute_topk(queries, corpus):
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred, truth):
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def time_queries(idx, queries, nprobe, early_stop):
    for q in queries[:5]:
        idx.search(q, k=KK, nprobe=nprobe, early_stop=early_stop)
    times: list[float] = []
    pred: list[list[int]] = []
    for q in queries:
        t0 = perf_counter()
        hits = idx.search(q, k=KK, nprobe=nprobe, early_stop=early_stop)
        times.append(perf_counter() - t0)
        pred.append([h[0] for h in hits])
    times.sort()
    return pred, float(np.mean(times) * 1e3), float(times[int(0.95 * len(times))] * 1e3)


def main() -> None:
    corpus = np.load(find_corpus()).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:500]

    print(f"corpus  : {corpus.shape}", flush=True)
    print(f"queries : {queries.shape}", flush=True)
    print(f"\nbrute-force ground truth ({len(queries)} × {len(corpus)})…",
          flush=True)
    t0 = perf_counter()
    truth = brute_topk(queries, corpus)
    print(f"  done in {perf_counter() - t0:.2f}s", flush=True)

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    print(f"\nfit({len(corpus)}) + add_batch…", flush=True)
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"\n{'nprobe':>6}  "
          f"{'recall (full)':>13}  {'ms full':>9}  "
          f"{'recall (early)':>14}  {'ms early':>9}  "
          f"{'speedup':>7}  {'Δrecall':>8}", flush=True)
    print("-" * 90, flush=True)
    for nprobe in NPROBES:
        pred_full, ms_full, _ = time_queries(idx, queries, nprobe, early_stop=False)
        pred_early, ms_early, _ = time_queries(idx, queries, nprobe, early_stop=True)
        r_full = recall_at_k(pred_full, truth)
        r_early = recall_at_k(pred_early, truth)
        speedup = ms_full / ms_early if ms_early > 0 else 0.0
        d_recall = r_early - r_full
        print(f"{nprobe:>6}  {r_full:>13.3f}  {ms_full:>9.2f}  "
              f"{r_early:>14.3f}  {ms_early:>9.2f}  "
              f"{speedup:>6.2f}×  {d_recall:>+8.3f}", flush=True)


if __name__ == "__main__":
    main()

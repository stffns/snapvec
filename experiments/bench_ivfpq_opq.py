"""OPQ vs non-OPQ recall comparison on BEIR FIQA.

Builds two IVFPQSnapIndex instances with identical hyperparameters
(nlist, M, K, nprobe) -- one with use_opq=True, one with the default
use_opq=False -- and reports recall@10 and latency for each.  The
delta isolates the effect of the learned rotation.
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
K = 256
KK = 10
NPROBES = [4, 8, 16, 32, 64, 128, 256]
SEED = 0
# Sweep M: larger d_sub = more room for OPQ variance balancing.
# Literature gains show up at d_sub >= 4.
M_VALUES = [48, 96, 192]


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def brute_topk(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred: list[list[int]], truth: np.ndarray) -> float:
    hits = sum(
        len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth)
    )
    return hits / (len(pred) * KK)


def time_queries(
    idx: IVFPQSnapIndex, queries: np.ndarray, nprobe: int,
) -> tuple[list[list[int]], float]:
    for q in queries[:5]:
        idx.search(q, k=KK, nprobe=nprobe)
    times: list[float] = []
    pred: list[list[int]] = []
    for q in queries:
        t0 = perf_counter()
        hits = idx.search(q, k=KK, nprobe=nprobe)
        times.append(perf_counter() - t0)
        pred.append([h[0] for h in hits])
    return pred, float(np.median(times) * 1e6)  # microseconds


def build(corpus: np.ndarray, M: int, use_opq: bool) -> IVFPQSnapIndex:
    dim = corpus.shape[1]
    idx = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=M, K=K,
        normalized=True, use_opq=use_opq, seed=SEED,
    )
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    print(f"  M={M} use_opq={use_opq}: fit + add_batch in "
          f"{perf_counter() - t0:.1f}s", flush=True)
    return idx


def main() -> None:
    corpus = np.load(find_corpus()).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:300]

    print(f"corpus  : {corpus.shape}", flush=True)
    print(f"queries : {queries.shape}", flush=True)
    print("\nbrute-force ground truth...", flush=True)
    truth = brute_topk(queries, corpus)
    print()

    for M in M_VALUES:
        print(f"\n=== M={M} (d_sub={corpus.shape[1] // M}) ===", flush=True)
        idx_base = build(corpus, M, use_opq=False)
        idx_opq = build(corpus, M, use_opq=True)

        print(
            f"\n{'nprobe':>6}  {'recall base':>11}  {'recall OPQ':>10}  "
            f"{'Δ recall':>9}  {'p50 base us':>11}  {'p50 OPQ us':>10}",
            flush=True,
        )
        print("-" * 70, flush=True)
        for nprobe in NPROBES:
            pred_b, us_b = time_queries(idx_base, queries, nprobe)
            pred_o, us_o = time_queries(idx_opq, queries, nprobe)
            r_b = recall_at_k(pred_b, truth)
            r_o = recall_at_k(pred_o, truth)
            delta = r_o - r_b
            print(
                f"{nprobe:>6}  {r_b:>11.3f}  {r_o:>10.3f}  "
                f"{delta:>+9.3f}  {us_b:>11.0f}  {us_o:>10.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

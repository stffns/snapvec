"""Baseline profile of IVFPQSnapIndex at N=1M.

This is the reference run that every later v0.5 perf optimisation is
measured against.  Reports:

  - recall@10 vs exact float32 brute-force on a held-out query batch
  - ms/query (mean and p95) at nprobe ∈ {32, 64, 128, 256}
  - cProfile breakdown of search() to identify the dominant cost
    (gather, LUT build, coarse matmul, top-k merge, Python overhead)

We augment the cached BGE-small / SciFact embeddings with small
Gaussian jitter on the unit sphere — same approach as
``bench_ivf_pq_contiguous.py`` — to reach N=1M without an embedding
pass that would block the sprint on a slow CPU pipeline.

Run: ``python experiments/bench_ivf_pq_1m_baseline.py``

Heads-up: N=1M build is ~1.5 GB RAM for the corpus and ~5–10 min for
the coarse k-means at nlist=4096.  Skip the brute-force ground truth
on a 200-query batch (~76 GB of ops in float32) is the slowest single
step — adjust ``N_QUERIES`` if running on a tight machine.
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from snapvec import IVFPQSnapIndex


# ──────────────────────────────────────────────────────────────────── #
# Config                                                                #
# ──────────────────────────────────────────────────────────────────── #

CACHE = Path("experiments/.cache_scifact_bge_small.npy")
N_CORPUS = 1_000_000
N_QUERIES = 200
N_TRAIN = 50_000
NLIST = 4096
M = 192
K = 256
KK = 10  # top-k
NPROBES = [32, 64, 128, 256]
SEED = 0
JITTER_SIGMA = 0.015


# ──────────────────────────────────────────────────────────────────── #
# Augmented corpus                                                      #
# ──────────────────────────────────────────────────────────────────── #

def augment(base: NDArray[np.float32], target_n: int, sigma: float, seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    reps = (target_n + len(base) - 1) // len(base)
    out = np.tile(base, (reps, 1)) + sigma * rng.standard_normal(
        (reps * len(base), base.shape[1])
    ).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    rng.shuffle(out)
    return out[:target_n]


def build_split() -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]
]:
    if not CACHE.exists():
        raise SystemExit(
            f"missing {CACHE}; run experiments/bench_multi_round_rht_real.py "
            "first to build the BGE-small / SciFact embedding cache."
        )
    base = np.load(CACHE).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12

    print(f"augmenting cache ({base.shape[0]} → {N_CORPUS + N_QUERIES + N_TRAIN}) "
          f"with σ={JITTER_SIGMA} jitter…")
    t0 = perf_counter()
    total = N_CORPUS + N_QUERIES + N_TRAIN
    full = augment(base, total, sigma=JITTER_SIGMA, seed=SEED)
    print(f"  done in {perf_counter() - t0:.1f}s, full corpus shape {full.shape}")

    corpus = full[:N_CORPUS]
    queries = full[N_CORPUS:N_CORPUS + N_QUERIES]
    train = full[N_CORPUS + N_QUERIES:]

    print(f"computing exact float32 brute-force ground truth "
          f"({N_QUERIES} × {N_CORPUS} × {corpus.shape[1]}d)…")
    t0 = perf_counter()
    # Chunked dot product to avoid materialising a 200 × 1M matrix at full.
    truth = np.empty((N_QUERIES, KK), dtype=np.int64)
    chunk = 50_000
    for q_start in range(0, N_QUERIES, 16):
        q_end = min(q_start + 16, N_QUERIES)
        q_block = queries[q_start:q_end]
        scores_block = np.full((q_end - q_start, KK), -np.inf, dtype=np.float32)
        ids_block = np.full((q_end - q_start, KK), -1, dtype=np.int64)
        for c_start in range(0, N_CORPUS, chunk):
            c_end = min(c_start + chunk, N_CORPUS)
            sub = q_block @ corpus[c_start:c_end].T
            # merge top-K into the running buffer
            cand_scores = np.concatenate([scores_block, sub], axis=1)
            cand_ids = np.concatenate(
                [ids_block, np.broadcast_to(
                    np.arange(c_start, c_end, dtype=np.int64),
                    (q_end - q_start, c_end - c_start)
                )], axis=1,
            )
            top_idx = np.argpartition(-cand_scores, KK - 1, axis=1)[:, :KK]
            scores_block = np.take_along_axis(cand_scores, top_idx, axis=1)
            ids_block = np.take_along_axis(cand_ids, top_idx, axis=1)
        # final sort within the top-KK
        order = np.argsort(-scores_block, axis=1)
        truth[q_start:q_end] = np.take_along_axis(ids_block, order, axis=1)
    print(f"  done in {perf_counter() - t0:.1f}s")

    return corpus, queries, train, truth


# ──────────────────────────────────────────────────────────────────── #
# Bench harness                                                         #
# ──────────────────────────────────────────────────────────────────── #

def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64]) -> float:
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def time_queries(idx: IVFPQSnapIndex, queries: NDArray[np.float32], nprobe: int) -> tuple[list[list[int]], float, float]:
    # warm-up
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
    mean_ms = float(np.mean(times) * 1e3)
    p95_ms = float(times[int(0.95 * len(times))] * 1e3)
    return pred, mean_ms, p95_ms


def profile_search(idx: IVFPQSnapIndex, queries: NDArray[np.float32], nprobe: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    for q in queries:
        idx.search(q, k=KK, nprobe=nprobe)
    profiler.disable()
    out = StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(25)
    return out.getvalue()


# ──────────────────────────────────────────────────────────────────── #
# Main                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    corpus, queries, train, truth = build_split()
    d = corpus.shape[1]
    print(f"\n=== IVFPQSnapIndex baseline @ N={N_CORPUS} d={d} "
          f"nlist={NLIST} M={M} K={K} ===")

    print(f"\nfit() on {N_TRAIN} training vectors…")
    t0 = perf_counter()
    idx = IVFPQSnapIndex(
        dim=d, nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    idx.fit(train, kmeans_iters=10)
    print(f"  done in {perf_counter() - t0:.1f}s")

    print(f"add_batch({N_CORPUS} vectors)…")
    t0 = perf_counter()
    idx.add_batch(list(range(N_CORPUS)), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s")

    sizes = np.diff(idx._offsets)
    print(f"  cluster sizes: min={int(sizes.min())}, "
          f"median={int(np.median(sizes))}, "
          f"max={int(sizes.max())}, "
          f"empty={int((sizes == 0).sum())}")

    print(f"\n{'nprobe':>6}  {'visited':>10}  {'recall@10':>10}  "
          f"{'mean ms':>8}  {'p95 ms':>8}")
    print("-" * 55)
    for nprobe in NPROBES:
        pred, mean_ms, p95_ms = time_queries(idx, queries, nprobe)
        r = recall_at_k(pred, truth)
        # Visited = sum of probed cluster sizes for the first query, as a
        # rough indicator (all queries probe the same nprobe count).
        visited = int(sizes.mean() * nprobe)
        print(f"{nprobe:>6}  {visited:>10}  {r:>10.3f}  "
              f"{mean_ms:>8.2f}  {p95_ms:>8.2f}")

    # Profile at the nprobe that targets ~0.94 recall, expected ≥ 64.
    profile_nprobe = NPROBES[1] if len(NPROBES) > 1 else NPROBES[0]
    print(f"\n=== cProfile breakdown @ nprobe={profile_nprobe} "
          f"({N_QUERIES} queries) ===")
    print(profile_search(idx, queries, profile_nprobe))


if __name__ == "__main__":
    main()

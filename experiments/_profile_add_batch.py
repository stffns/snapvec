"""Quick profile of IVFPQSnapIndex.add_batch at N=100k.

Confirms which sub-operation dominates the 14-min cost we measured at
N=1M.  Runs in ~30s on a laptop so we can iterate fixes without
waiting for the full N=1M cycle.
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np

from snapvec import IVFPQSnapIndex


CACHE = Path("experiments/.cache_scifact_bge_small.npy")
N_CORPUS = 100_000
N_TRAIN = 50_000
NLIST = 1024  # n_train / nlist = 50 — healthy cluster size
M = 192
K = 256


def augment(base, target_n, sigma=0.015, seed=0):
    rng = np.random.default_rng(seed)
    reps = (target_n + len(base) - 1) // len(base)
    out = np.tile(base, (reps, 1)) + sigma * rng.standard_normal(
        (reps * len(base), base.shape[1])
    ).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    rng.shuffle(out)
    return out[:target_n]


def main() -> None:
    base = np.load(CACHE).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    full = augment(base, N_CORPUS + N_TRAIN, sigma=0.015, seed=0)
    corpus, train = full[:N_CORPUS], full[N_CORPUS:]

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=0,
    )
    print(f"fit() on {N_TRAIN} training vectors…")
    t0 = perf_counter()
    idx.fit(train, kmeans_iters=10)
    print(f"  done in {perf_counter() - t0:.1f}s")

    print(f"\nadd_batch({N_CORPUS}) — single call ── timing")
    ids = list(range(N_CORPUS))

    t0 = perf_counter()
    idx.add_batch(ids, corpus)
    wall = perf_counter() - t0
    print(f"  total wall: {wall:.2f}s ({wall * 1000 / N_CORPUS:.3f} ms/vec)")

    # Re-create + profile
    idx2 = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=0,
    )
    idx2.fit(train, kmeans_iters=10)

    print(f"\nadd_batch({N_CORPUS}) — cProfile breakdown")
    profiler = cProfile.Profile()
    profiler.enable()
    idx2.add_batch(ids, corpus)
    profiler.disable()
    out = StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(20)
    print(out.getvalue())


if __name__ == "__main__":
    main()

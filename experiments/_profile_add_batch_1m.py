"""Profile add_batch at N=1M to confirm whether chunking fixes the
catastrophic 14-min cost we saw in bench_ivf_pq_1m_baseline."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from snapvec import IVFPQSnapIndex


CACHE = Path("experiments/.cache_scifact_bge_small.npy")
N_CORPUS = 1_000_000
N_TRAIN = 200_000
NLIST = 4096
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
    print(f"augmenting → {N_CORPUS + N_TRAIN}…", flush=True)
    t0 = perf_counter()
    full = augment(base, N_CORPUS + N_TRAIN, sigma=0.015, seed=0)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)
    corpus, train = full[:N_CORPUS], full[N_CORPUS:]

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=0,
    )
    print(f"fit() on {N_TRAIN} training vectors (n_train/nlist = "
          f"{N_TRAIN // NLIST})…", flush=True)
    t0 = perf_counter()
    idx.fit(train, kmeans_iters=10)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"\nadd_batch({N_CORPUS}) — single call", flush=True)
    ids = list(range(N_CORPUS))
    t0 = perf_counter()
    idx.add_batch(ids, corpus)
    wall = perf_counter() - t0
    print(f"  total wall: {wall:.2f}s ({wall * 1000 / N_CORPUS:.3f} ms/vec)",
          flush=True)

    sizes = np.diff(idx._offsets)
    print(f"  cluster sizes: min={int(sizes.min())}, "
          f"median={int(np.median(sizes))}, "
          f"max={int(sizes.max())}, "
          f"empty={int((sizes == 0).sum())}", flush=True)


if __name__ == "__main__":
    main()

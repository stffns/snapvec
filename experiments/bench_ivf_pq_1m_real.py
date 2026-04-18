"""N=1M baseline using REAL BGE-small embeddings (FIQA seed).

Same harness as ``bench_ivf_pq_1m_v2.py`` but the seed corpus is the
57 638 FIQA documents embedded on Colab Pro / A100, not the 3 300
SciFact docs.  At N=1M that is ~18 jitter copies per real seed
instead of ~320 → near-duplicate density drops by ~18×, which should
unstick the recall-saturated-at-0.758 ceiling we saw with
``bench_ivf_pq_1m_v2.py``.

Same n_train=200 000 and nlist=4096 (n_train / nlist = 48), so any
recall gain is attributable to the underlying corpus diversity, not
to the IVF-PQ configuration.

Run: ``python experiments/bench_ivf_pq_1m_real.py``
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from snapvec import IVFPQSnapIndex


CACHE = Path("experiments/.cache_fiqa_bge_small.npy")
N_CORPUS = 1_000_000
N_QUERIES = 200
N_TRAIN = 200_000
NLIST = 4096
M = 192
K = 256
KK = 10
NPROBES = [16, 32, 64, 128, 256]
SEED = 0
JITTER_SIGMA = 0.015


def augment(base, target_n, sigma=JITTER_SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    reps = (target_n + len(base) - 1) // len(base)
    out = np.tile(base, (reps, 1)) + sigma * rng.standard_normal(
        (reps * len(base), base.shape[1])
    ).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    rng.shuffle(out)
    return out[:target_n]


def brute_topk(queries, corpus):
    truth = np.empty((len(queries), KK), dtype=np.int64)
    chunk = 50_000
    for q_start in range(0, len(queries), 16):
        q_end = min(q_start + 16, len(queries))
        q_block = queries[q_start:q_end]
        scores_block = np.full((q_end - q_start, KK), -np.inf, dtype=np.float32)
        ids_block = np.full((q_end - q_start, KK), -1, dtype=np.int64)
        for c_start in range(0, len(corpus), chunk):
            c_end = min(c_start + chunk, len(corpus))
            sub = q_block @ corpus[c_start:c_end].T
            cand_scores = np.concatenate([scores_block, sub], axis=1)
            cand_ids = np.concatenate(
                [ids_block, np.broadcast_to(
                    np.arange(c_start, c_end, dtype=np.int64),
                    (q_end - q_start, c_end - c_start)
                )], axis=1,
            )
            top = np.argpartition(-cand_scores, KK - 1, axis=1)[:, :KK]
            scores_block = np.take_along_axis(cand_scores, top, axis=1)
            ids_block = np.take_along_axis(cand_ids, top, axis=1)
        order = np.argsort(-scores_block, axis=1)
        truth[q_start:q_end] = np.take_along_axis(ids_block, order, axis=1)
    return truth


def recall_at_k(pred, truth):
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def time_queries(idx, queries, nprobe):
    for q in queries[:5]:
        idx.search(q, k=KK, nprobe=nprobe)
    times = []
    pred = []
    for q in queries:
        t0 = perf_counter()
        hits = idx.search(q, k=KK, nprobe=nprobe)
        times.append(perf_counter() - t0)
        pred.append([h[0] for h in hits])
    times.sort()
    return pred, float(np.mean(times) * 1e3), float(times[int(0.95 * len(times))] * 1e3)


def main() -> None:
    base = np.load(CACHE).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    print(f"seed corpus: FIQA / BGE-small, {base.shape}", flush=True)

    print(f"augmenting → {N_CORPUS + N_QUERIES + N_TRAIN} "
          f"({(N_CORPUS + N_QUERIES + N_TRAIN) // len(base)} reps per seed)",
          flush=True)
    t0 = perf_counter()
    full = augment(base, N_CORPUS + N_QUERIES + N_TRAIN)
    corpus = full[:N_CORPUS]
    queries = full[N_CORPUS:N_CORPUS + N_QUERIES]
    train = full[N_CORPUS + N_QUERIES:]
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"\nbrute-force ground truth ({N_QUERIES} × {N_CORPUS})…", flush=True)
    t0 = perf_counter()
    truth = brute_topk(queries, corpus)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"\n=== IVFPQSnapIndex N={N_CORPUS} d={corpus.shape[1]} "
          f"nlist={NLIST} M={M} K={K} ===", flush=True)
    print(f"n_train = {N_TRAIN} (n_train / nlist = {N_TRAIN // NLIST})",
          flush=True)

    idx = IVFPQSnapIndex(
        dim=corpus.shape[1], nlist=NLIST, M=M, K=K,
        normalized=True, seed=SEED,
    )
    print(f"\nfit({N_TRAIN})…", flush=True)
    t0 = perf_counter()
    idx.fit(train, kmeans_iters=10)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    print(f"add_batch({N_CORPUS})…", flush=True)
    t0 = perf_counter()
    idx.add_batch(list(range(N_CORPUS)), corpus)
    print(f"  done in {perf_counter() - t0:.1f}s", flush=True)

    sizes = np.diff(idx._offsets)
    print(f"\ncluster sizes: min={int(sizes.min())} median={int(np.median(sizes))} "
          f"max={int(sizes.max())} empty={int((sizes == 0).sum())}", flush=True)

    print(f"\n{'nprobe':>6}  {'recall@10':>10}  {'mean ms':>9}  {'p95 ms':>9}",
          flush=True)
    print("-" * 45, flush=True)
    for nprobe in NPROBES:
        pred, mean_ms, p95_ms = time_queries(idx, queries, nprobe)
        r = recall_at_k(pred, truth)
        print(f"{nprobe:>6}  {r:>10.3f}  {mean_ms:>9.2f}  {p95_ms:>9.2f}",
              flush=True)


if __name__ == "__main__":
    main()

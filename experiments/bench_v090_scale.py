"""v0.9.0 scale benchmark: N=57K, 200K, 500K, 1M.

Augments the FIQA corpus with jittered copies to reach target N.
Measures: recall@10, latency (us/query), build time, disk size.
"""
from __future__ import annotations

import os
import tempfile
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

M = 192
K = 256
KK = 10
N_QUERIES = 200
SEED = 0
NPROBE_SWEEP = [16, 32, 64, 128]

SCALE_CONFIGS = [
    # (target_N, nlist, n_train)
    (57_638, 512, None),       # original FIQA, no augmentation
    (200_000, 1024, 50_000),
    (500_000, 2048, 100_000),
    (1_000_000, 4096, 200_000),
]


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("Missing FIQA corpus .npy")


def augment_corpus(
    base: NDArray[np.float32], target_n: int, rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Create jittered copies to reach target_n."""
    if target_n <= len(base):
        return base[:target_n]
    copies_needed = target_n - len(base)
    indices = rng.integers(0, len(base), size=copies_needed)
    jitter = rng.normal(0, 0.01, size=(copies_needed, base.shape[1])).astype(np.float32)
    augmented = base[indices] + jitter
    augmented /= np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-12
    return np.vstack([base, augmented])


def brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32],
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64]) -> float:
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * KK)


def bench_scale(
    corpus: NDArray[np.float32],
    queries: NDArray[np.float32],
    truth: NDArray[np.int64],
    nlist: int,
    n_train: int | None,
) -> None:
    N = len(corpus)
    dim = corpus.shape[1]
    train_n = n_train or N

    ivf = IVFPQSnapIndex(
        dim=dim, nlist=nlist, M=M, K=K,
        normalized=True, keep_full_precision=True, seed=SEED,
    )

    # Fit
    t0 = perf_counter()
    ivf.fit(corpus[:train_n], kmeans_iters=15)
    fit_s = perf_counter() - t0

    # Add
    t0 = perf_counter()
    ivf.add_batch(list(range(N)), corpus)
    add_s = perf_counter() - t0

    # Disk size
    with tempfile.NamedTemporaryFile(suffix=".snpi", delete=False) as f:
        tmp_path = f.name
    ivf.save(tmp_path)
    disk_mb = os.path.getsize(tmp_path) / 1024 / 1024
    os.unlink(tmp_path)

    stats = ivf.stats()
    sizes = np.diff(ivf._offsets)

    print(f"  fit:     {fit_s:.1f}s")
    print(f"  add:     {add_s:.1f}s")
    print(f"  disk:    {disk_mb:.1f} MB ({stats['bytes_per_vec']} B/vec, "
          f"{stats.get('bytes_per_vec_codes_only', '?')} codes-only)")
    print(f"  clusters: min={int(sizes.min())} median={int(np.median(sizes))} "
          f"max={int(sizes.max())} empty={int((sizes == 0).sum())}")

    # Warm up
    for q in queries[:5]:
        ivf.search(q, k=KK, nprobe=32)

    print(f"\n  {'nprobe':>6}  {'recall@10':>10}  {'us/query':>10}"
          f"  {'rerank rec':>10}  {'rerank us':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for nprobe in NPROBE_SWEEP:
        if nprobe > nlist:
            continue

        pred_pq = []
        t0 = perf_counter()
        for q in queries:
            hits = ivf.search(q, k=KK, nprobe=nprobe)
            pred_pq.append([h[0] for h in hits])
        us_pq = (perf_counter() - t0) / len(queries) * 1e6
        rec_pq = recall_at_k(pred_pq, truth)

        pred_rr = []
        t0 = perf_counter()
        for q in queries:
            hits = ivf.search(q, k=KK, nprobe=nprobe, rerank_candidates=100)
            pred_rr.append([h[0] for h in hits])
        us_rr = (perf_counter() - t0) / len(queries) * 1e6
        rec_rr = recall_at_k(pred_rr, truth)

        print(f"  {nprobe:>6}  {rec_pq:>10.3f}  {us_pq:>10.0f}"
              f"  {rec_rr:>10.3f}  {us_rr:>10.0f}")


def main() -> None:
    corpus_path = find_corpus()
    if not QUERIES_PATH.exists():
        raise SystemExit(f"Missing queries .npy at {QUERIES_PATH}")

    base_corpus = np.load(corpus_path).astype(np.float32)
    base_corpus /= np.linalg.norm(base_corpus, axis=1, keepdims=True) + 1e-12
    queries = np.load(QUERIES_PATH).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:N_QUERIES]

    rng = np.random.default_rng(SEED)

    print("snapvec v0.9.0 -- Scale Benchmark")
    print(f"Base corpus: {base_corpus.shape[0]:,} x {base_corpus.shape[1]}")
    print(f"Queries: {len(queries)}")

    for target_n, nlist, n_train in SCALE_CONFIGS:
        print(f"\n\n{'#'*70}")
        print(f"# N = {target_n:,}  |  nlist = {nlist}  |  M={M}  K={K}")
        print(f"{'#'*70}")

        corpus = augment_corpus(base_corpus, target_n, rng)
        print(f"  corpus: {corpus.shape[0]:,} vectors")

        print(f"\n  Brute-force ground truth...", flush=True)
        t0 = perf_counter()
        truth = brute_topk(queries, corpus)
        bf_s = perf_counter() - t0
        print(f"  done in {bf_s:.1f}s")

        bench_scale(corpus, queries, truth, nlist, n_train)


if __name__ == "__main__":
    main()

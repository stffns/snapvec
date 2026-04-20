"""Competitive ANN comparison on BEIR FIQA.

Head-to-head between snapvec (``IVFPQSnapIndex``), FAISS (``IndexIVFPQ``),
and hnswlib at comparable accuracy operating points.  Reports p50 and
p99 latency per single-query call (not batched), recall@10 vs
brute-force float32, and on-disk footprint.

Each backend runs in a fresh subprocess so the three competing libomp
builds that get linked into a single Python process on macOS (FAISS,
hnswlib, and snapvec all bundle their own) don't collide at runtime.
The parent collects JSON per backend and prints a combined table.

Usage (orchestrator)::

    python experiments/bench_competitive.py

Usage (single backend, called by the orchestrator)::

    python experiments/bench_competitive.py --backend snapvec --out snapvec.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


CORPUS_CANDIDATES = [
    Path("experiments/.cache_fiqa_corpus_bge_small.npy"),
    Path("experiments/.cache_fiqa_bge_small.npy"),
]
QUERIES_PATH = Path("experiments/.cache_fiqa_queries_bge_small.npy")

K = 10
N_QUERY_SAMPLE = 200
NLIST = 512
NPROBE = 32
FAISS_M = 48               # 384 / 48 = 8, FAISS requires dim % M == 0
SNAPVEC_M = 192            # snapvec's FIQA operating point
PQ_K = 256
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def load_fiqa() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    corpus = np.load(find_corpus()).astype(np.float32)
    queries = np.load(QUERIES_PATH).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries[:N_QUERY_SAMPLE]
    return corpus, queries


def brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32], k: int,
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def recall_at_k(
    pred: NDArray[np.int64], truth: NDArray[np.int64], k: int,
) -> float:
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(pred) * k)


def time_per_query(fn, queries: NDArray[np.float32]) -> tuple[float, float]:
    """p50 and p99 latency in microseconds."""
    for q in queries[:5]:
        fn(q)
    gc.collect()
    gc.disable()
    try:
        times = []
        for q in queries:
            t0 = perf_counter()
            fn(q)
            times.append(perf_counter() - t0)
    finally:
        gc.enable()
    arr = np.array(times) * 1e6
    return float(np.median(arr)), float(np.percentile(arr, 99))


def disk_size(save_fn) -> int:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = Path(f.name)
    try:
        save_fn(path)
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# per-backend runners                                                          #
# --------------------------------------------------------------------------- #


def run_snapvec() -> list[dict]:
    from snapvec import IVFPQSnapIndex, __version__ as snapvec_version

    corpus, queries = load_fiqa()
    truth = brute_topk(queries, corpus, K)
    dim = corpus.shape[1]
    results = []

    # flagship: rerank on
    idx = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=SNAPVEC_M, K=PQ_K,
        normalized=True, keep_full_precision=True, seed=0,
    )
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    build = perf_counter() - t0
    pred = np.array([
        [i for i, _ in idx.search(q, k=K, nprobe=NPROBE, rerank_candidates=100)]
        for q in queries
    ])
    p50, p99 = time_per_query(
        lambda q: idx.search(q, k=K, nprobe=NPROBE, rerank_candidates=100),
        queries,
    )
    disk = disk_size(lambda p: idx.save(p))
    idx.close()
    results.append(dict(
        name=f"snapvec IVFPQ + fp16 rerank (M={SNAPVEC_M}, nprobe={NPROBE})",
        recall=recall_at_k(pred, truth, K),
        p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
        notes="fp16 rerank cache",
        library_version=snapvec_version,
    ))

    # pure PQ
    idx2 = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=SNAPVEC_M, K=PQ_K,
        normalized=True, seed=0,
    )
    t0 = perf_counter()
    idx2.fit(corpus, kmeans_iters=15)
    idx2.add_batch(list(range(len(corpus))), corpus)
    build2 = perf_counter() - t0
    pred2 = np.array([
        [i for i, _ in idx2.search(q, k=K, nprobe=NPROBE)] for q in queries
    ])
    p50_2, p99_2 = time_per_query(
        lambda q: idx2.search(q, k=K, nprobe=NPROBE), queries,
    )
    disk2 = disk_size(lambda p: idx2.save(p))
    idx2.close()
    results.append(dict(
        name=f"snapvec IVFPQ no rerank (M={SNAPVEC_M}, nprobe={NPROBE})",
        recall=recall_at_k(pred2, truth, K),
        p50_us=p50_2, p99_us=p99_2, disk_bytes=disk2, build_s=build2,
        notes="PQ only",
        library_version=snapvec_version,
    ))
    return results


def run_faiss() -> list[dict]:
    import faiss
    faiss.omp_set_num_threads(1)

    corpus, queries = load_fiqa()
    truth = brute_topk(queries, corpus, K)
    dim = corpus.shape[1]

    results = []

    for m in (FAISS_M, SNAPVEC_M):
        quantizer = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFPQ(quantizer, dim, NLIST, m, 8)
        idx.nprobe = NPROBE
        t0 = perf_counter()
        idx.train(corpus)
        idx.add(corpus)
        build = perf_counter() - t0

        pred = np.empty((len(queries), K), dtype=np.int64)
        for i, q in enumerate(queries):
            _, res = idx.search(q.reshape(1, -1), K)
            pred[i] = res[0]
        p50, p99 = time_per_query(
            lambda q: idx.search(q.reshape(1, -1), K), queries,
        )
        disk = disk_size(lambda p: faiss.write_index(idx, str(p)))
        results.append(dict(
            name=f"FAISS IVFPQ (M={m}, nprobe={NPROBE})",
            recall=recall_at_k(pred, truth, K),
            p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
            notes="8-bit PQ, no rerank"
                  + (" [matched-budget]" if m == SNAPVEC_M else ""),
            library_version=faiss.__version__,
        ))

    return results


def run_hnswlib() -> list[dict]:
    from importlib.metadata import version as _pkg_version
    import hnswlib

    corpus, queries = load_fiqa()
    truth = brute_topk(queries, corpus, K)
    dim = corpus.shape[1]

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(
        max_elements=len(corpus),
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
        random_seed=0,
    )
    t0 = perf_counter()
    idx.add_items(corpus, np.arange(len(corpus)))
    idx.set_ef(HNSW_EF_SEARCH)
    idx.set_num_threads(1)
    build = perf_counter() - t0

    pred = np.empty((len(queries), K), dtype=np.int64)
    for i, q in enumerate(queries):
        labels, _ = idx.knn_query(q.reshape(1, -1), k=K)
        pred[i] = labels[0]
    p50, p99 = time_per_query(
        lambda q: idx.knn_query(q.reshape(1, -1), k=K), queries,
    )
    disk = disk_size(lambda p: idx.save_index(str(p)))

    try:
        hnswlib_version = _pkg_version("hnswlib")
    except Exception:
        hnswlib_version = "unknown"
    return [dict(
        name=f"hnswlib (M={HNSW_M}, ef_search={HNSW_EF_SEARCH})",
        recall=recall_at_k(pred, truth, K),
        p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
        notes="no train phase",
        library_version=hnswlib_version,
    )]


# --------------------------------------------------------------------------- #
# orchestrator                                                                 #
# --------------------------------------------------------------------------- #


def run_backend_subprocess(backend: str) -> list[dict]:
    """Invoke this same file in a fresh subprocess for a single backend.

    Each library bundles its own libomp on macOS; dynamically linking
    two of them into the same process crashes on Apple Silicon.  Running
    in clean child processes keeps each backend's runtime happy.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        subprocess.run(
            [
                sys.executable, __file__,
                "--backend", backend, "--out", out_path,
            ],
            check=True,
        )
        with open(out_path) as f:
            return json.load(f)
    finally:
        os.unlink(out_path)


def main_orchestrator() -> None:
    print(f"Hardware: {platform.platform()}")
    print(f"Python {platform.python_version()}  numpy {np.__version__}")
    print()

    all_rows: list[dict] = []
    for backend in ("snapvec", "faiss", "hnswlib"):
        print(f"running backend: {backend} (subprocess)...")
        all_rows.extend(run_backend_subprocess(backend))

    print()
    hdr = (
        f"{'backend':<55}  {'recall@10':>9}  "
        f"{'p50 us':>7}  {'p99 us':>7}  {'disk MB':>7}  "
        f"{'build s':>7}  notes"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(
            f"{r['name']:<55}  {r['recall']:>9.3f}  "
            f"{r['p50_us']:>7.1f}  {r['p99_us']:>7.1f}  "
            f"{r['disk_bytes'] / 1e6:>7.1f}  {r['build_s']:>7.1f}  "
            f"{r['notes']}"
        )


def main_backend(backend: str, out_path: str) -> None:
    runner = {
        "snapvec": run_snapvec,
        "faiss": run_faiss,
        "hnswlib": run_hnswlib,
    }[backend]
    rows = runner()
    with open(out_path, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("snapvec", "faiss", "hnswlib"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.backend is not None and args.out is not None:
        main_backend(args.backend, args.out)
    else:
        main_orchestrator()

"""HNSW-over-PQ spike: does HNSW traversal on PQ codes beat IVFPQ+rerank?

Question this spike answers: if we swapped IVF probing for HNSW
traversal on the same PQ-encoded codes, would recall/latency on
FIQA move materially enough to justify a v0.12 flagship rework?

Baselines (reproduced for apples-to-apples):
  A. snapvec IVFPQ M=192, no rerank           -- current matched-
                                                 budget vs FAISS
  B. snapvec IVFPQ M=192 + fp16 rerank        -- current flagship
  C. snapvec IVFPQ M=48 + OPQ                 -- aggressive corner
  D. FAISS IndexHNSWPQ M=192                  -- hypothesis, M=192
  E. FAISS IndexHNSWPQ M=48                   -- hypothesis, M=48

Each backend runs in a fresh subprocess because faiss, hnswlib, and
snapvec all bundle their own libomp on macOS arm64.  Parent reads
JSON per backend and prints a combined table.

FAISS note: `IndexHNSWPQ` is L2-only at the C++ layer (the factory
and the metric_type setter both silently stay on L2 for the HNSW
family).  This is fine here because unit-normalised vectors give
`|a-b|^2 = 2 - 2<a,b>`, so L2 top-k equals IP top-k in ranking;
rows are labelled L2 honestly.

Usage::

    python experiments/spike_hnswpq.py                          # all
    python experiments/spike_hnswpq.py --backend faiss_hnswpq \
        --out /tmp/faiss.json                                   # one

Results (2026-04-22, BEIR FIQA N=57,638 dim=384, M4 Pro, single-thread):

  snapvec IVFPQ + fp16 rerank (M=192)      0.945  352us   54.3 MB
  FAISS HNSWPQ L2 (M=192, ef=256)          0.936  644us   25.9 MB
  FAISS HNSWPQ L2 (M=192, ef=128)          0.934  412us   25.9 MB
  FAISS HNSWPQ L2 (M=192, ef=64)           0.926  281us   25.9 MB
  snapvec IVFPQ no rerank (M=192)          0.895  305us   12.1 MB
  snapvec IVFPQ + OPQ no rerank (M=48)     0.649  253us    4.7 MB
  FAISS HNSWPQ L2 (M=48, ef=128)           0.603  133us   18.0 MB
  FAISS HNSWPQ L2 (M=48, ef=256)           0.602  216us   18.0 MB
  FAISS HNSWPQ L2 (M=48, ef=64)            0.585   91us   18.0 MB

Verdict: hypothesis rejected.

  1. Recall ceiling of HNSW+PQ at M=192 is ~0.934-0.936 on this
     corpus, still 1 pp below snapvec's rerank flagship (0.945) and
     6 pp below hnswlib-fp32 (0.994).  The ceiling is a function of
     PQ codebook resolution (M, K, data distribution), not of the
     traversal strategy -- switching IVF -> HNSW does not close it.
  2. Disk is NOT small.  HNSW edge list (graph_M * N * 4 bytes) at
     graph_M=32 is 7.0 MB just for the edges, larger than the PQ
     codes for this N.  FAISS HNSWPQ = 25.9 MB vs snapvec IVFPQ
     no-rerank = 12.1 MB.  Ratio stays ~2x at higher N; the "small
     disk + graph traversal" pitch is physically inconsistent.
  3. At M=48 the aggressive-compression corner, snapvec+OPQ beats
     FAISS HNSWPQ on recall (0.649 vs 0.603, +4.6 pp) -- same
     pattern as the existing IVFPQ-vs-FAISS-IVFPQ comparison in
     the paper.  HNSW does not change the high-compression story.
  4. Only new Pareto point: FAISS HNSWPQ M=192 ef=64 at 0.926 /
     281us / 25.9 MB -- better recall than snapvec no-rerank
     (0.895) at similar latency, but 2x disk.  Sideways move, not
     a frontier push.

Decision: defer HNSW+PQ.  v0.12 flagship goes to mmap + delta
buffer (the scale-ceiling story), which is the only axis where
snapvec still has headroom on this workload.
"""
from __future__ import annotations

import argparse
import gc
import json
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
PQ_K = 256
SNAPVEC_M_FLAGSHIP = 192
SNAPVEC_M_SMALL = 48

HNSW_GRAPH_M = 32
HNSW_EF_CONSTRUCTION = 200
EF_SEARCH_SWEEP = (64, 128, 256)


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing FIQA corpus cache (run _colab_embed_corpus.py).")


def load_fiqa() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    corpus = np.load(find_corpus()).astype(np.float32, copy=False)
    queries = np.load(QUERIES_PATH).astype(np.float32, copy=False)
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    queries = queries[:N_QUERY_SAMPLE].astype(np.float32, copy=False)
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
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
# backends                                                                    #
# --------------------------------------------------------------------------- #


def run_snapvec() -> list[dict]:
    from snapvec import IVFPQSnapIndex, __version__ as snapvec_version

    corpus, queries = load_fiqa()
    truth = brute_topk(queries, corpus, K)
    dim = corpus.shape[1]
    results: list[dict] = []

    # A. matched-budget, no rerank
    idx = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=SNAPVEC_M_FLAGSHIP, K=PQ_K,
        normalized=True, seed=0,
    )
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    build = perf_counter() - t0
    pred = np.array([
        [i for i, _ in idx.search(q, k=K, nprobe=NPROBE)]
        for q in queries
    ])
    p50, p99 = time_per_query(
        lambda q: idx.search(q, k=K, nprobe=NPROBE), queries,
    )
    disk = disk_size(lambda p: idx.save(p))
    idx.close()
    results.append(dict(
        name=f"snapvec IVFPQ no rerank (M={SNAPVEC_M_FLAGSHIP})",
        recall=recall_at_k(pred, truth, K),
        p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
        notes=f"nprobe={NPROBE}",
        library_version=snapvec_version,
    ))

    # B. flagship with rerank
    idx = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=SNAPVEC_M_FLAGSHIP, K=PQ_K,
        normalized=True, keep_full_precision=True, seed=0,
    )
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    build = perf_counter() - t0
    pred = np.array([
        [
            i for i, _ in idx.search(
                q, k=K, nprobe=NPROBE, rerank_candidates=100,
            )
        ]
        for q in queries
    ])
    p50, p99 = time_per_query(
        lambda q: idx.search(
            q, k=K, nprobe=NPROBE, rerank_candidates=100,
        ),
        queries,
    )
    disk = disk_size(lambda p: idx.save(p))
    idx.close()
    results.append(dict(
        name=f"snapvec IVFPQ + fp16 rerank (M={SNAPVEC_M_FLAGSHIP})",
        recall=recall_at_k(pred, truth, K),
        p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
        notes=f"nprobe={NPROBE}, rerank=100",
        library_version=snapvec_version,
    ))

    # C. aggressive corner with OPQ
    idx = IVFPQSnapIndex(
        dim=dim, nlist=NLIST, M=SNAPVEC_M_SMALL, K=PQ_K,
        normalized=True, use_opq=True, seed=0,
    )
    t0 = perf_counter()
    idx.fit(corpus, kmeans_iters=15)
    idx.add_batch(list(range(len(corpus))), corpus)
    build = perf_counter() - t0
    pred = np.array([
        [i for i, _ in idx.search(q, k=K, nprobe=NPROBE)]
        for q in queries
    ])
    p50, p99 = time_per_query(
        lambda q: idx.search(q, k=K, nprobe=NPROBE), queries,
    )
    disk = disk_size(lambda p: idx.save(p))
    idx.close()
    results.append(dict(
        name=f"snapvec IVFPQ + OPQ no rerank (M={SNAPVEC_M_SMALL})",
        recall=recall_at_k(pred, truth, K),
        p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
        notes=f"nprobe={NPROBE}",
        library_version=snapvec_version,
    ))

    return results


def run_faiss_hnswpq() -> list[dict]:
    import faiss  # type: ignore[import-not-found]

    faiss.omp_set_num_threads(1)

    corpus, queries = load_fiqa()
    truth = brute_topk(queries, corpus, K)
    dim = corpus.shape[1]
    results: list[dict] = []

    # FAISS IndexHNSWPQ is L2-only at the C++ layer (the factory and
    # the Python metric_type setter both silently stay on L2 for HNSW
    # family indices).  That is fine for this spike: the vectors are
    # unit-normalised, so |a-b|^2 = 2 - 2<a,b> and L2 top-k equals IP
    # top-k in ranking.  PQ codebooks are trained to minimise L2
    # reconstruction error, which is also the right objective for IP
    # approximation on normalised queries.  Label rows as L2 honestly.
    for m in (SNAPVEC_M_FLAGSHIP, SNAPVEC_M_SMALL):
        if dim % m != 0:
            continue
        idx = faiss.IndexHNSWPQ(dim, m, HNSW_GRAPH_M)
        assert idx.metric_type == faiss.METRIC_L2, (
            "expected L2; FAISS HNSWPQ does not support IP at build"
        )
        idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        t0 = perf_counter()
        idx.train(corpus)
        idx.add(corpus)
        build = perf_counter() - t0

        for ef in EF_SEARCH_SWEEP:
            idx.hnsw.efSearch = ef

            # Time recall and latency on the same 1-query-at-a-time
            # path so they describe the same execution.  Earlier
            # version used a batch call for pred and a per-query call
            # for latency -- different code paths, different timings.
            preds = []
            for q in queries[:5]:
                idx.search(q.reshape(1, -1), K)
            gc.collect()
            gc.disable()
            try:
                times = []
                for q in queries:
                    t0q = perf_counter()
                    _, I = idx.search(q.reshape(1, -1), K)
                    times.append(perf_counter() - t0q)
                    preds.append(I[0])
            finally:
                gc.enable()
            arr = np.array(times) * 1e6
            p50 = float(np.median(arr))
            p99 = float(np.percentile(arr, 99))
            pred = np.array(preds, dtype=np.int64)

            disk = disk_size(
                lambda p: faiss.write_index(idx, str(p)),  # noqa: B023
            )
            results.append(dict(
                name=f"FAISS IndexHNSWPQ L2 (M={m}, ef={ef})",
                recall=recall_at_k(pred, truth, K),
                p50_us=p50, p99_us=p99, disk_bytes=disk, build_s=build,
                notes=(
                    f"hnsw_M={HNSW_GRAPH_M}, efC={HNSW_EF_CONSTRUCTION}, "
                    f"efSearch={ef}, L2==IP ranking on unit vectors"
                ),
                library_version=faiss.__version__,
            ))

    return results


BACKENDS = {
    "snapvec": run_snapvec,
    "faiss_hnswpq": run_faiss_hnswpq,
}


# --------------------------------------------------------------------------- #
# orchestrator                                                                #
# --------------------------------------------------------------------------- #


def _fmt_row(r: dict) -> str:
    disk_mb = r["disk_bytes"] / (1024 * 1024)
    return (
        f"{r['name']:<52}  {r['recall']:>7.3f}  "
        f"{r['p50_us']:>6.0f}  {r['p99_us']:>6.0f}  "
        f"{disk_mb:>6.2f}  {r['build_s']:>6.1f}"
    )


def orchestrate() -> None:
    rows: list[dict] = []
    for backend in BACKENDS:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json",
        ) as tmp:
            out = Path(tmp.name)
        try:
            subprocess.run(
                [
                    sys.executable, __file__,
                    "--backend", backend, "--out", str(out),
                ],
                check=True,
            )
            rows.extend(json.loads(out.read_text()))
        finally:
            out.unlink(missing_ok=True)

    rows.sort(key=lambda r: (-r["recall"], r["p50_us"]))
    header = (
        f"{'backend':<52}  {'recall':>7}  {'p50us':>6}  "
        f"{'p99us':>6}  {'MB':>6}  {'build':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(_fmt_row(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.backend is None:
        orchestrate()
        return

    results = BACKENDS[args.backend]()
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(_fmt_row(r))


if __name__ == "__main__":
    main()

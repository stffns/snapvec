# Benchmarks

All numbers measured on the same hardware with a fixed random seed. Raw
scripts live in [`experiments/`](https://github.com/stffns/snapvec/tree/main/experiments);
a reproducible CI-runnable suite is planned.

## Headline: IVFPQSnapIndex on FIQA

BEIR FIQA, N = 57,638, dim = 384 (BGE-small), nlist = 512, M = 192, K = 256.
`keep_full_precision=True`, `rerank_candidates=100`.

| `nprobe` | Configuration | recall@10 | Latency (us/query) |
|---------|---------------|-----------|--------------------|
| 8 | PQ only | 0.85 | 180 |
| 32 | PQ only | 0.92 | 340 |
| 64 | PQ + fp16 rerank | **0.977** | **441** |
| 256 | PQ + fp16 rerank | **0.998** | **1021** |

5.8x speedup vs v0.6 at identical recall, past the PQ-only 0.929 ceiling.

## snapvec vs sqlite-vec (measured)

Same corpus, same hardware. `snapvec` at `nprobe=64` + rerank.

| N | sqlite-vec | snapvec | Speedup | Recall tradeoff |
|---|------------|---------|---------|-----------------|
| 10k | 2.3 ms | 0.44 ms | 5x | 0.997 |
| 57k | 15.1 ms | 0.44 ms | 34x | 0.977 |
| 100k | 23.8 ms | 1.04 ms | 23x | 0.994 |
| 500k | ~110 ms | 0.9 ms | 125x | ~0.97 |
| 1M | brute-force infeasible | 1.1 ms | -- | -- |

Disk footprint is 2-8x smaller across the range.

## snapvec vs FAISS vs hnswlib (measured)

Head-to-head on BEIR FIQA (N = 57,638, dim = 384, BGE-small),
Apple M4 Pro, single-thread per-query latency, p50 and p99 over 200
queries, 10-NN vs brute-force float32 ground truth.

| Backend | recall@10 | p50 us | p99 us | disk MB | build s |
|---------|----------:|-------:|-------:|--------:|--------:|
| `snapvec` IVFPQ + fp16 rerank (M=192) | **0.945** | **342** | 474 | 56.9 | 128 |
| `snapvec` IVFPQ no rerank (M=192)     | 0.895 | 352 | 658 | 12.6 | 127 |
| FAISS IVFPQ (M=192) [matched-budget]  | 0.906 | 476 | 551 | 12.7 | 17 |
| FAISS IVFPQ (M=48)                    | 0.603 | 145 | 189 | 4.4 | 10 |
| hnswlib (M=32, ef_search=128)         | **0.994** | 601 | 1033 | 104.5 | 8 |

All backends pinned to a single thread for per-query latency parity
(`faiss.omp_set_num_threads(1)`, `hnswlib.set_num_threads(1)`).

Matched-budget line (same PQ M=192, same no-rerank config):

- snapvec 0.895 recall @ 352 us
- FAISS   0.906 recall @ 476 us (**1.35x slower**, essentially the same recall and disk)

With `fp16 rerank`, snapvec lifts recall to 0.945 without extra
latency (342 us p50) at the cost of 4.5x disk (fp16 cache stored next
to the PQ codes).  hnswlib reaches higher recall (0.994) but at 2-3x
latency and 2x disk vs snapvec+rerank.

Where each wins:

- **snapvec**: best single-thread latency at high recall; lowest disk
  under matched PQ; zero runtime deps beyond NumPy.
- **FAISS**: much faster `fit` (16 s vs 128 s for snapvec at M=192)
  and the smallest footprint when PQ compression is pushed hard.
- **hnswlib**: highest ceiling recall if disk and latency budget are
  flexible; no training phase.

Reproduce with `python experiments/bench_competitive.py` after
caching the FIQA corpus + queries.

## Batched search threading curve

`IVFPQSnapIndex.search_batch` fans out per-query scoring across worker
threads.  Threading is a throughput knob, not a per-call latency knob:
the single `search()` API is serial on purpose (it competes with
NumPy's internal BLAS pool; see the docstring).

Same FIQA corpus as above (N = 57,638, dim = 384), batch_size = 128,
measured on an Apple M4 Pro (12 cores), NumPy 2.4.3, Python 3.12.

| `nprobe` | t=1 ms/q | t=2 ms/q | t=4 ms/q | t=8 ms/q | best speedup |
|---------:|---------:|---------:|---------:|---------:|:------------:|
| 4   | 0.09 | 0.06 | **0.05** | 0.06 | 1.69x |
| 8   | 0.15 | 0.09 | **0.07** | 0.09 | 2.31x |
| 16  | 0.29 | 0.16 | **0.10** | 0.13 | 2.92x |
| 32  | 0.50 | 0.28 | **0.16** | 0.20 | 3.06x |
| 64  | 0.95 | 0.51 | **0.29** | 0.33 | 3.31x |
| 128 | 1.84 | 0.98 | **0.54** | 0.57 | 3.43x |
| 256 | 3.70 | 1.91 | **1.01** | 1.09 | 3.67x |

Observations:

- **`num_threads=4` is the sweet spot** on this machine across every
  nprobe.  At `num_threads=8` the curve regresses; the executor
  over-subscribes the efficiency cores and starts fighting BLAS.
- **Scaling improves with `nprobe`** because per-query work grows:
  at `nprobe=4`, threading overhead caps speedup at 1.7x; at
  `nprobe=256` it reaches 3.7x.
- **Sub-millisecond at 4 threads** for `nprobe <= 64`, which spans
  the 0.85 to 0.977 recall range from the headline table above.
  That is 3,400 - 20,000 queries per second per process.

Reproduce with `python experiments/bench_ivf_pq_threading.py` after
caching both the FIQA corpus (`experiments/.cache_fiqa_bge_small.npy`)
and the FIQA queries (`experiments/.cache_fiqa_queries_bge_small.npy`).

## Compression ratios

For BGE-small (dim=384, float32 baseline = 1536 B/vec):

| Index | Config | B/vec | Compression |
|-------|--------|-------|-------------|
| `SnapIndex` | bits=2 | 132 | 11.6x |
| `SnapIndex` | bits=3 | 196 | 7.8x |
| `SnapIndex` | bits=4 | 260 | 5.9x |
| `PQSnapIndex` | M=16 | 16 | 96x |
| `PQSnapIndex` | M=32 | 32 | 48x |
| `IVFPQSnapIndex` | M=192 + fp16 rerank | ~960 | 1.6x (rerank cache dominates) |
| `IVFPQSnapIndex` | M=192, no rerank | ~192 | 8x |

## Reproduction

```bash
pip install -e ".[dev]"
python experiments/bench_v090_fiqa.py               # FIQA recall / latency
python experiments/bench_ivf_pq_threading.py        # search_batch threading curve
python experiments/bench_sqlite_vec_baseline.py     # sqlite-vec comparison
```

The `experiments/` folder is WIP; expect rough edges. A first-class
`bench/` suite that runs in CI and emits machine-readable results is
tracked on the roadmap.

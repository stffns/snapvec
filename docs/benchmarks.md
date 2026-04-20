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

## Head-to-head: snapvec vs FAISS vs hnswlib vs sqlite-vec

Single unified table across every backend tested, at their standard
operating points **and** at matched-budget points where we have a
comparable PQ config.  Goal is to let the reader see the full Pareto
shape rather than cherry-picked wins.

**Config.** BEIR FIQA corpus (N = 57,638, dim = 384, BGE-small,
unit-normalised).  200 queries sampled from FIQA's 648 test queries.
Apple M4 Pro, 12 cores, 24 GB RAM, NumPy 2.4.3, Python 3.12.  Every
backend pinned to a single thread for apples-to-apples per-query
latency.  GC disabled inside the timing loop.  **Ground truth** is
float32 brute-force top-10 dot product on the same unit-normalised
corpus -- recall@10 is against exact NN.

Every row below is approximate (recall < 1.0) **except sqlite-vec**,
which is an exact brute-force scan.  snapvec has no exact mode --
every `SnapIndex` / `PQSnapIndex` / `IVFPQSnapIndex` variant quantizes
the vectors on ingest, so the lowest-tier `SnapIndex 4-bit` row is
"full-scan + scalar quantization", **not** flat-exact in the FAISS
`IndexFlat` sense.  For flat-exact behaviour with snapvec's storage
story, use sqlite-vec or FAISS `IndexFlatIP`.

| Backend | recall@10 | p50 us | p99 us | disk MB | build s |
|---------|----------:|-------:|-------:|--------:|--------:|
| sqlite-vec (brute-force cosine, exact) | **1.000** | 13891 | 18628 | 91.1 | 0.5 |
| hnswlib (M=32, ef_search=128) | 0.994 | 561 | 994 | 104.5 | 45 |
| **snapvec IVFPQ + fp16 rerank (M=192)** | **0.945** | **359** | 457 | 56.9 | 108 |
| FAISS IVFPQ (M=192) [matched-budget] | 0.906 | 483 | 584 | 12.7 | 17 |
| **snapvec IVFPQ no rerank (M=192)** | 0.895 | **325** | 376 | 12.6 | 110 |
| snapvec SnapIndex 4-bit scalar (full-scan) | 0.854 | 2676 | 3164 | 15.4 | 1.1 |
| snapvec SnapIndex 3-bit scalar (full-scan) | 0.736 | 2688 | 3013 | 11.7 | 0.8 |
| snapvec SnapIndex 2-bit scalar (full-scan) | 0.618 | 2726 | 4016 | 8.0 | 0.7 |
| FAISS IVFPQ (M=48) | 0.603 | 142 | 200 | 4.4 | 10 |
| snapvec IVFPQ no rerank (M=48) [matched-budget] | 0.549 | 267 | 350 | 4.3 | 33 |

Rows ordered by recall@10 descending.

**Methodology.** All numbers come from a single end-to-end invocation
of `experiments/bench_competitive.py`.  The orchestrator spawns one
subprocess per backend (each backend bundles its own libomp; loading
two into the same Python process crashes on macOS arm64) but all four
subprocesses run back-to-back in one session so OS page-cache state
does not drift between earlier and later rows.  An earlier
multi-session measurement showed a ~56% p50 delta on SnapIndex between
cold and warm runs; the single-invocation convention above eliminates
that.

Thread pinning: `faiss.omp_set_num_threads(1)` + `idx.set_num_threads(1)`
on the hnswlib instance (set before `add_items`) for apples-to-apples
build and search timings.  snapvec's `fit` still uses whatever NumPy
BLAS is configured to; for this machine `np.show_config()` reports
Accelerate with its default thread count.  Expected run-to-run noise
on this hardware is <=5% p50 for every row except hnswlib, which
hits ~10-15% routinely.

### Reading the table

The Pareto frontier (no backend strictly dominated) is:

1. **FAISS IVFPQ M=48** owns the aggressive-compression corner --
   0.603 recall at 4.4 MB and 142 us.  snapvec at the same M budget
   is slower AND has lower recall, so at that ultra-compressed point
   FAISS wins outright.
2. **snapvec IVFPQ M=192** matches FAISS M=192 on disk (12.6 vs
   12.7 MB) and on recall (0.895 vs 0.906) while being **1.5x faster**
   at p50 (325 vs 483 us).  This is the matched-budget headline.
3. **snapvec IVFPQ + fp16 rerank** is the Pareto-dominant high-recall
   point under 500 us: 0.945 recall at 359 us -- faster than FAISS
   M=192 AND higher recall, at the cost of a 4.5x larger index file
   (holds a float16 copy for the rerank pass).
4. **hnswlib** reaches the highest non-exact recall (0.994) but pays
   with disk (104 MB) and p99 latency (994 us).
5. **sqlite-vec** is exact (recall 1.000) but its brute-force
   cosine scan is 13.9 ms -- ~40x slower than any of the ANN backends
   on this N.  It's the 'zero ANN tuning, accept the latency' baseline.

### Positioning in plain language

- If you need **one dependency, no training, acceptable latency on
  small N**: `SnapIndex` at 4-bit scalar or sqlite-vec.  snapvec is
  ~5x faster (2.7 ms vs 13.9 ms) but gives up exactness (0.85 vs
  1.00 recall) because it quantizes the vectors.  `SnapIndex` p50
  latency is essentially constant across bit depths (the fp16
  centroid-expansion matmul dominates); the recall/disk tradeoff is
  the only knob you turn.
- If you have **space for PQ training and want aggressive disk
  compression (~4 MB)**: FAISS IVFPQ M=48 is the winner at that
  point on this hardware.
- If you want **matched disk and the fastest ANN latency for
  recall ~0.9**: snapvec IVFPQ M=192.
- If you want **recall approaching 0.95 in sub-millisecond latency**:
  snapvec IVFPQ + fp16 rerank (4.5x disk for ~5 pp recall lift).
- If you want **the highest recall regardless of disk / tail
  latency**: hnswlib.

### Caveats

- FAISS `fit` is ~6.5x faster than snapvec's `fit` at the same config
  (17 s vs 110 s at M=192).  Build time is a real competitive gap.
- FAISS IVFPQ at M=48 beats snapvec at M=48; snapvec's PQ training
  isn't uniformly better at every compression point.  The advantage
  shows up at mid-range (M=96-192).
- These are serial per-query numbers.  hnswlib in particular gets a
  large speedup from its default thread pool; the [threading curve]
  section covers how snapvec scales batched search.

Reproduce with `python experiments/bench_competitive.py` after
caching both `experiments/.cache_fiqa_bge_small.npy` and
`experiments/.cache_fiqa_queries_bge_small.npy`.

## Historical: snapvec vs sqlite-vec across N

The earlier (pre-competitive-table) snapvec-vs-sqlite-vec scale
measurement, kept for the absolute scaling story snapvec unlocks.
`snapvec` at `nprobe=64` + rerank.

| N | sqlite-vec | snapvec | Speedup | Recall tradeoff |
|---|------------|---------|---------|-----------------|
| 10k | 2.3 ms | 0.44 ms | 5x | 0.997 |
| 57k | 15.1 ms | 0.44 ms | 34x | 0.977 |
| 100k | 23.8 ms | 1.04 ms | 23x | 0.994 |
| 500k | ~110 ms | 0.9 ms | 125x | ~0.97 |
| 1M | brute-force infeasible | 1.1 ms | -- | -- |

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

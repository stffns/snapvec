# Performance Notes — `snapvec` v0.5 Sprint

Negative-result archive for techniques tried during the
"push NumPy to its limits" perf sprint, with measured numbers and
the diagnosis for why each one didn't work.  Kept here so we don't
spend the budget twice on ideas that already failed.

## 2026-04-15 — Early-stop / short-circuit probing on `IVFPQSnapIndex`

**Idea.**  Walk the probed clusters in descending coarse-score order,
maintain a running top-k threshold, break as soon as no remaining
cluster could possibly beat the threshold.  The upper bound for any
unseen cluster `c` is `coarse_dot[c] + Σⱼ max(lut[j])`.  Standard
trick in C/C++ ANN libraries (FAISS, ScaNN).

**A/B (FIQA queries vs FIQA corpus, N=57 638, nlist=512, M=192, K=256):**

| nprobe | recall full | ms full | recall early | ms early | speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.661 | 0.46 | 0.661 | 0.85 | **0.54×** |
| 8 | 0.771 | 0.58 | 0.771 | 1.47 | 0.40× |
| 16 | 0.845 | 0.94 | 0.845 | 2.74 | 0.34× |
| 32 | 0.891 | 1.55 | 0.891 | 5.21 | 0.30× |
| 64 | 0.914 | 2.70 | 0.914 | 10.07 | 0.27× |
| 128 | 0.924 | 4.96 | 0.924 | 19.92 | 0.25× |
| 256 | 0.929 | 9.57 | 0.929 | 39.64 | 0.24× |

**Δrecall is exactly 0.000 in every row** — the upper bound is too loose
to skip anything.  Yet early_stop is **2–4× slower than the default
search path**.

**Why it loses in NumPy:**

1. The full-search path does *one* per-subspace gather over the
   concatenated probed-cluster slab — **`M = 192` total NumPy dispatch
   calls**.
2. The early-stop path does the same gather *per cluster* —
   **`nprobe × M` NumPy dispatch calls** (~12k for nprobe = 64).
3. NumPy's per-call dispatch cost (~2 µs) dominates at this granularity.
4. The C/C++ implementations of the same trick avoid this because
   their per-cluster work is just a tight loop, not a Python interpreter
   round-trip per subspace.

**The upper bound itself is also loose:** `Σⱼ max(lut[j])` assumes the
query lands on the best codeword in *every* subspace independently —
unrealistic.  Real per-cluster maxima are typically half of this bound
or less, so even after sorting clusters by coarse score, almost no
cluster's UB falls below the running top-k threshold within the first
few visited clusters.

**Verdict.**  Not viable in pure-NumPy.  Re-evaluate if/when we add a
C/Cython hot path or move the per-cluster loop into a vectorised
`numba` kernel.  Tracked as v0.6+ follow-up.

## 2026-04-15 — `ThreadPoolExecutor` parallel probe in `IVFPQSnapIndex.search`

**Idea.**  Split probed clusters into N chunks, fan the per-cluster
gather + scoring out across worker threads.  NumPy releases the GIL on
``np.take`` and contiguous-array arithmetic, so the parallelism is
real.  Expected 3-4× speedup on a 4-core laptop.

**A/B (FIQA queries vs FIQA corpus, N=57 638, nlist=512, M=192,
K=256), num_threads ∈ {1, 2, 4, 8}:**

| nprobe | t=1 | t=2 | t=4 | t=8 | best speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.49 | 1.02 | 0.47 | 0.47 | 1.05× (noise) |
| 8 | 0.61 | 1.39 | 3.11 | 0.61 | 0.99× |
| 16 | 0.97 | 1.82 | 4.15 | 7.07 | **0.53×** (slower!) |
| 32 | 1.61 | 2.83 | 6.71 | 10.31 | 0.57× |
| 64 | 2.86 | 3.94 | 7.81 | 15.42 | 0.73× |
| 128 | 5.21 | 6.72 | 9.51 | 17.30 | 0.78× |
| 256 | 9.77 | **6.50** | 13.10 | 19.74 | **1.50×** (only win) |

**Verdict.**  Threading at the single-query level is mostly slower or
noise.  Only at nprobe = 256 with t=2 do we see a real 1.5× — at the
typical operating point for "0.91 recall" (nprobe = 64 at this corpus
size) it lands at 0.73×.

**Why it loses:**

1. **Apple Silicon Accelerate / Intel MKL already parallelise**
   ``np.take`` and matmul internally.  Worker threads compete with
   the BLAS thread pool for the same physical cores.
2. **Thread-spawn dispatch overhead** is ~200-500 µs per chunk —
   significant when total work is < 5 ms.
3. The operations being parallelised (the per-subspace gather + sum)
   are already vectorised in C inside NumPy.  We add Python-level
   coordination on top of work that was already running in parallel.

**Where threading would actually pay** is at the **batch level** —
process N independent queries in parallel, where each query has
substantial per-thread work and the Python overhead is amortised
across the batch.  That is the design for ``search_batch`` (task #6).

**Verdict.**  Drop ``num_threads`` from ``search()`` for v0.5.  Move
the threading concept into ``search_batch`` where the workload shape
makes it fly.

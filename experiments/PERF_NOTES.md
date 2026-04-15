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

## 2026-04-15 — SWAR / quantised LUT for the residual ADC inner loop

**Idea.**  Replace the float32 per-subspace LUT with a quantised int8
representation, accumulate scores in int16, dequantise once at the
end.  Variants:

1. **Per-subspace int8 LUT, float32 accumulator.**  Each subspace
   stores ``(bias_j, scale_j, lut_q[j, :] uint8)``; per-iteration
   ``scores += scale_j * lut_q[j][cat[:, j]].astype(float32)``.
2. **Pure int16 accumulator with averaged scale.**  Idealised
   (mathematically incorrect because per-subspace scales differ) but
   useful as an upper-bound on what the int-only path could buy.
3. **Pair-LUT SWAR.**  Pack two consecutive subspace codes per
   uint16 entry, precompute ``lut_pair[p, c1·256 + c2]`` int16, and
   reduce the inner loop from ``M`` iterations to ``M/2``.

**Microbench (synthetic shapes mirroring the FIQA bench:
total = 3500, M = 192, K = 256, 200 runs):**

| variant | ms / call | speedup |
|---|---:|---:|
| baseline (float32 LUT) | 1.120 | 1.00× |
| per-subspace int8 LUT | 1.480 | **0.76× (slower)** |
| pure-int idealised | 1.325 | 0.85× (still slower) |

Pair-LUT SWAR was not microbenched because **its construction cost
alone is prohibitive at K = 256**: a per-query
``(M/2, K²) int16`` table is ``96 × 65536 × 2 = 12 MB``, written
from scratch every search.  At ~5 GB/s effective DRAM bandwidth on
this laptop, that is ~2.4 ms of build cost — already more than the
**entire** baseline search latency at the recall sweep's typical
operating points.  And the table is too big to reuse across queries
in a batch since it is query-dependent.

**Why every variant loses in NumPy:**

1. The hot path at M = 192 is **dispatch-bound, not memory-bound**.
   Each per-subspace iteration is a single NumPy call with ~2 µs of
   Python + dispatch overhead.  M iterations → ~400 µs of dispatch
   floor regardless of what the per-element work costs.
2. Reducing per-element bytes (int8 vs float32) does not help when
   the bottleneck is iteration count.
3. The "fused gather + add" for ``scores += lut[j][cat[:, j]]`` is
   already a single tight C inner loop inside NumPy — adding a cast
   or multiply per iteration just buys more dispatch.
4. SWAR gains hinge on building a wide pair LUT, which at K = 256 is
   too big to materialise per query.  At K = 16 it would fit in L1
   but our deployed K is 256 for the recall it buys.

**Verdict.**  Not viable in pure NumPy at our operating M and K.
Re-evaluate when (if) we drop into a Numba / Cython / C kernel that
fuses the per-subspace ops into a single SIMD loop — at that point
the dispatch floor disappears and the int / SWAR pipelines become
real wins.  Tracked as part of the v0.6+ "optional accelerator"
roadmap.

## Aggregate finding — the NumPy ceiling for IVF-PQ search

Combining the negative results above, the search hot path in pure
NumPy at M = 192, K = 256 has a **structural floor of ~5 µs per
visited candidate** — about 1.5 ms / query at the FIQA "0.91 recall"
operating point (nprobe = 32 visiting ~3500 candidates).

What works in pure NumPy (shipping in v0.5):
- Column-major (M, n) code storage: **~12 % latency win** on the
  per-subspace gather.
- Chunked encoding in ``add_batch``: **12× build-time speedup at
  N = 1M** (memory pressure was the actual bottleneck).
- Batched coarse + LUT in ``search_batch``: ergonomic API,
  performance-neutral vs per-query loop in pure NumPy.

What does *not* work in pure NumPy:
- Algorithmic short-circuit (early_stop): dispatch overhead.
- Single-query thread parallelism: BLAS pool already parallel.
- Quantised / SWAR LUT: dispatch-bound, not memory-bound.

Reaching < 1 ms / query at the 0.91-recall operating point requires
either dropping to a compiled inner loop (Numba / Cython / C) or
reducing M significantly (which costs recall).  This is the v0.5
ceiling, and the v0.6 roadmap.

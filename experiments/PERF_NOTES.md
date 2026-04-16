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


## 2026-04-16 -- Vectorised LUT + ADC Sprint

Goal: eliminate the two M=192 Python loops in the search hot path
(LUT construction and ADC scoring) without leaving pure NumPy.

### LUT construction: batched matmul -- SHIPPED (7x)

Replace M separate per-subspace matmuls with one batched np.matmul:

```python
# Before: 192 dispatches, ~140 us
for j in range(M):
    lut[j] = codebooks[j] @ q_split[j]

# After: 1 dispatch, ~19 us
q_split = q_pre.reshape(M, d_sub, 1)
lut = np.matmul(codebooks, q_split).squeeze(-1)
```

Also replaced einsum with transpose+matmul in the batch path.

**Measured:** 7.3x speedup (140 us -> 19 us), consistent across runs.
After this change, LUT build drops from ~10% of search time to <1%.

### ADC scoring: vectorised approaches -- ALL FAILED

Tested five strategies to replace the per-subspace scoring loop:

| Method                        | n=500  | n=3.5K | n=50K  |
|-------------------------------|--------|--------|--------|
| loop (baseline, col-major)    | 1.00x  | 1.00x  | 1.00x  |
| np.take_along_axis            | 0.85x  | 0.43x  | 0.43x  |
| fancy indexing                | 0.85x  | 0.43x  | 0.43x  |
| chunked (8/16/32/64)         | 0.61x  | 0.38x  | 0.40x  |
| flat indexing (precomputed)   | 2.14x  | 1.12x  | 0.96x  |
| sparse CSR matvec (no build)  | 3.29x  | 1.71x  | 1.48x  |
| sparse CSR matvec (with build)| 0.38x  | 0.20x  | 0.20x  |

**Why they lose:** All vectorised approaches materialise an (M, n_cand)
intermediate (2.7 MB at n=3500). The loop accumulates in-place into a
14 KB buffer that stays in L1 cache. Cache locality beats dispatch
elimination at every realistic operating point.

**Sparse CSR** wins spectacularly when the matrix is pre-built, but CSR
construction is 5-10x more expensive than the matvec itself, killing
the total.

**Flat precomputed** (store int32 flat codes at build time) gives 1.12x
at n=3500 but costs 4x RAM for codes -- not worth it.

### PQ column-major layout -- SHIPPED (1.47x)

PQSnapIndex stored codes as (n, M) row-major. The ADC loop accesses
one subspace at a time: `codes[:, j]` strides M bytes per element.
Switching to (M, n) makes each subspace access contiguous, matching
the IVF-PQ layout that already had this optimisation.

**Measured (N=20K, M=192):**
- Row-major (n, M): 5708 us
- Column-major (M, n): 3888 us -- **1.47x speedup**

File format unchanged (transpose on save/load for backward compat).

### End-to-end profile after all changes

**IVFPQSnapIndex** (N=20K, nprobe=32, ~11.7K candidates):

| Stage          |    us |     % |
|----------------|------:|------:|
| preprocess     |     6 |  0.2% |
| coarse_probe   |    19 |  0.6% |
| lut_build      |    25 |  0.8% |
| gather         |   676 | 21.6% |
| adc_loop       |  2315 | 73.9% |
| topk           |    84 |  2.7% |
| **TOTAL**      |  3134 |       |

**PQSnapIndex** (N=20K, full scan):

| Stage          |    us |     % |
|----------------|------:|------:|
| preprocess     |    10 |  0.2% |
| lut_build      |    23 |  0.4% |
| adc_loop       |  5673 | 97.0% |
| topk           |   159 |  2.7% |
| **TOTAL**      |  5850 |       |

### Conclusions

The ADC scoring loop is genuinely optimal in pure NumPy. The per-
subspace `scores += lut[j][cat[j]]` pattern has perfect L1 cache
locality (14 KB working set) and cannot be beaten by any approach
that materialises the full (M, n_cand) matrix.

The remaining bottleneck split is:
- **IVF-PQ:** 74% ADC + 22% gather. The gather phase (Python loop
  over probed clusters, slicing contiguous memory) is the next target
  but vectorising it requires knowing the cluster boundaries at
  compile time -- another dispatch-bound wall.
- **PQ:** 97% ADC. Nothing else matters.

### Gather phase vectorisation -- FAILED

The gather loop copies codes from nprobe probed clusters into a
contiguous (M, total) buffer. Tested four vectorisation strategies:

| Method                     | us   | vs baseline |
|----------------------------|-----:|------------:|
| loop (baseline)            |  104 |       1.00x |
| np.concatenate             |  114 |       0.92x |
| fancy index (codes[:,idx]) |  689 |       0.15x |
| cumsum (zero Python loops) |  686 |       0.15x |

**Why they lose:** The loop copies contiguous slices (memcpy per
cluster). Fancy indexing gathers scattered indices across 192 rows,
destroying cache locality. np.concatenate adds list-comprehension
and allocation overhead without eliminating the per-cluster work.

Baseline breakdown: 83% codes copy (contiguous, fast), 14% arange
for row_idx, 8% scores fill. No inefficiency to exploit.


## Pure-NumPy ceiling statement (2026-04-16)

Every stage of the search pipeline has been profiled and tested
against vectorised alternatives. The current implementation is
at or near the pure-NumPy optimum:

- **LUT build:** fused via batched matmul (7x win, shipped).
- **ADC scoring:** per-subspace loop with L1-resident accumulator
  beats all materialise-and-reduce strategies.
- **Gather:** contiguous-slice loop beats fancy indexing and
  concatenation.
- **Top-k / coarse probe / preprocess:** <4% combined, not worth
  optimising.

The path to < 1ms/query remains compiled code (Numba/Cython/Rust).

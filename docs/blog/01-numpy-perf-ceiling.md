# I tried 6 NumPy-level optimizations on a vector index. 4 lost. Here's the math.

*Draft — author: Jay (stffns) · 2026-04-15*

---

## TL;DR

I spent a sprint pushing the search hot path of an IVF-PQ vector
index ([`snapvec`](https://github.com/stffns/snapvec)) to its limit
in pure NumPy.  Six well-known optimizations on the table:

| Attempt | Outcome | Best ratio vs baseline |
|---|---|---|
| Column-major code storage | **win** | 1.12× |
| Chunked `add_batch` (build path) | **win** | 12× |
| Batched search API | **neutral** | 1.0× |
| Early-stop / short-circuit probing | **loss** | 0.24× |
| Single-query `ThreadPoolExecutor` | **loss** | 0.27× — 1.5× peak |
| Quantised int8 LUT / SWAR ADC | **loss** | 0.76× |

Everything that lost, lost for the **same structural reason** — and
that reason is not what the NumPy optimization folklore predicts.
Worth writing up because (a) the negative findings are rarely
documented and (b) the diagnosis tells you *where* the NumPy
vectorization story actually breaks.

---

## What I was optimising

`snapvec`'s `IVFPQSnapIndex.search()` is a textbook IVF-PQ probe:

1. Score each of `nlist` coarse k-means centroids against the query
   (one matmul, fast).
2. Pick the top `nprobe` clusters.
3. For each candidate vector in those clusters, sum its per-subspace
   contributions from a precomputed query LUT.

The hot path that consumes most of the wall time is step 3 — what
FAISS calls *asymmetric distance computation* (ADC).  In pure NumPy
it looks like this, repeated `M = 192` times per query:

```python
for j in range(M):
    scores += lut[j][cat[:, j]]
```

`cat` is a `(total, M)` uint8 array of PQ codes for the candidate
set; `lut` is `(M, K=256)` float32; `scores` is `(total,)` float32.
At the FIQA recall sweep's typical operating point, `total ≈ 3500`
and one `search()` call takes ~1.5 ms.

The goal was getting that down — ideally toward `< 1 ms / query` at
N = 1M and 0.94 recall, on a laptop CPU, in pure NumPy.

## The wins

### 1. Column-major code storage — `1.12×`

Stored `_codes` as `(M, n)` instead of `(n, M)` so that the
per-subspace gather `_codes[:, slice]` reads a contiguous strip of
memory instead of stride-`M=192` byte hops.  Cache-friendlier; no
recall change.  Modest 12% latency win, mostly from the gather phase.

### 2. Chunked `add_batch` — `12×` (build path)

The actual blowup was unrelated to search: `add_batch(N=1M)` was
taking **14 minutes**.  `cProfile` showed the function body itself
hot — meaning *NumPy ops*, not Python overhead.  Diagnosis: at
N=1M, the `residuals` matrix `(1M × 384)` is **1.5 GB float32**.
On an 8 GB laptop with browser + IDE running, that pushed peak
memory into swap.

Fix: encode in chunks of 65 536 vectors so peak transient memory
stays under ~150 MB.  827 s → 70 s, **12× faster**, no algorithmic
change.

This isn't really a NumPy optimization — it's "stop making the OS
swap" — but it's the single biggest wall-clock win of the sprint
and it's the kind of bug that's invisible until N is large enough
to see it.

### 3. `search_batch` — neutral, but still shipped

Built the per-batch coarse score matrix `(B, nlist)` and the LUT
tensor `(B, M, K)` with single BLAS calls instead of looping
`search()` per query.  Expectation: 5× throughput from amortising
the matmul setup costs.

Measured: **~1.0×**.  At any batch size from 1 to 500, batched is
within float noise of a per-query Python loop.

Why?  Apple Accelerate (and Intel MKL) already amortise BLAS calls
internally.  Even at B=8, the small matmul saturates available
parallelism.  The "single BLAS call per batch" win that motivates
batched APIs in C++ frameworks evaporates because NumPy's BLAS
pool was already getting it.

I shipped it anyway — the API is ergonomic for production batched
flows, and it'll start showing real wins once we add a Numba-
accelerated inner loop in v0.6 (where per-call dispatch matters
more).  But it was the first signal that the NumPy ceiling is
closer than I'd hoped.

## The losses

### 4. Early-stop probing — `0.24×` (4× *slower*)

Standard trick from FAISS / ScaNN: walk probed clusters in
descending coarse-score order; after each cluster, check if the
upper bound `coarse_dot[c] + Σⱼ max(lut[j])` for the next cluster
is below the current top-k threshold; if so, break.

Measured A/B at every nprobe: zero recall change (the upper bound
is too loose to skip anything) and **2-4× slower than the default
search path**.

The diagnosis is the structural one I'll keep coming back to:

- Default search path: **`M = 192` total NumPy dispatch calls** —
  one fused `scores += lut[j][cat[:, j]]` per subspace, vectorised
  over the whole concatenated candidate slab.
- Early-stop path: **`nprobe × M` dispatch calls** — same gather
  per cluster, ~12 000 dispatches at nprobe=64.

NumPy's per-call dispatch overhead is ~2 µs.  At 12 000 dispatches,
that's 24 ms of pure overhead — already 16× the entire baseline
search latency.  The C++ implementations of the same trick avoid
this because their per-cluster work is just a tight loop, not a
Python interpreter round-trip per subspace.

### 5. Single-query `ThreadPoolExecutor` — peak `1.5×` at one config, mostly worse

NumPy releases the GIL on `np.take` and contiguous-array
arithmetic, so I split the probed clusters into N chunks and
processed each chunk in a worker thread.  Expected 3-4× on a
4-core laptop.

Measured at FIQA, sweeping `nprobe ∈ {4, 8, 16, 32, 64, 128, 256}`
× `num_threads ∈ {1, 2, 4, 8}`:

| nprobe | t=1 | t=2 | t=4 | t=8 | best |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.97 | 1.82 | 4.15 | 7.07 | 0.53× |
| 32 | 1.61 | 2.83 | 6.71 | 10.31 | 0.57× |
| 64 | **2.86** | 3.94 | 7.81 | 15.42 | 0.73× |
| 128 | 5.21 | 6.72 | 9.51 | 17.30 | 0.78× |
| 256 | 9.77 | **6.50** | 13.10 | 19.74 | **1.50×** |

Only one configuration won: nprobe=256 with t=2, peaking at 1.5×.
Everything else lost.  At the typical operating point (nprobe=64,
recall 0.91), threading is **0.73×** — actively slower.

Two structural reasons:

1. **Apple Accelerate / Intel MKL already parallelise** `np.take`
   and matmul internally.  My worker threads competed with the
   BLAS pool for the same physical cores.
2. **Thread spawn / dispatch overhead** is ~200-500 µs per chunk.
   When the total work is < 5 ms, that's 10-20% of the budget.

The default Python advice — "release the GIL and parallelise" —
silently assumes you're competing with a single-threaded baseline.
You aren't, if you're calling a BLAS-backed NumPy.

### 6. Quantised int8 LUT / SWAR ADC — `0.76×`

Replace the float32 per-subspace LUT with int8, accumulate scores
in int16, dequantise once.  Memory traffic per gather is 4× lower;
inner-loop bytes per element drop from 4 to 1.  Should be a
1.3-1.5× win on bandwidth alone.

Microbench:

| variant | ms / call | speedup |
|---|---:|---:|
| baseline (float32 LUT) | 1.120 | 1.00× |
| per-subspace int8 LUT | 1.480 | **0.76×** |
| pure-int idealised | 1.325 | 0.85× |

Same structural reason as early-stop: the hot path is
**dispatch-bound, not memory-bound**.  Each per-subspace iteration
is ~2 µs of Python + NumPy dispatch.  M iterations means a
~400 µs floor regardless of what the per-element work costs.
Reducing per-element bytes does not move the needle when the
bottleneck is iteration count.  Adding a cast (`uint8 → float32`)
or a per-subspace multiply just buys *more* dispatch.

I didn't even microbench the full SWAR pair-LUT variant.  At
K=256 the per-query pair LUT is `(M/2, K²) int16 = 12 MB` to
write from scratch every search.  At ~5 GB/s effective DRAM
bandwidth, that's 2.4 ms of build cost — *more than the entire
baseline search latency*.  Some optimizations are dead at the
napkin-math stage.

## The pattern

Every loss has the same shape: **the per-subspace inner loop in
Python is too coarse-grained for NumPy's vectorisation story to
help**.

NumPy's pitch is "write Python code that calls C inner loops."
That works beautifully when each Python call corresponds to a
*lot* of C work — a matmul, a sum over a million elements, a
reduction across a tensor.  It breaks down when the per-call work
is small but happens many times.  At M=192 subspaces, even a
pristine vectorised inner per call (~5 µs) gets *dwarfed* by the
2 µs interpreter dispatch each call costs.  The C inside NumPy
isn't the bottleneck — the Python *outside* it is.

The conventional NumPy optimization advice — vectorise loops,
release the GIL, batch computations — is correct but **assumes the
inner loop has enough work to amortise the dispatch**.  At our M,
it doesn't.  And no amount of clever per-element trickery (int8,
SWAR, threading, early-stop) helps because none of them remove the
dispatch.

The only thing that removes the dispatch is **moving the per-
subspace loop inside C**.  In our case that means a Numba kernel
or a Cython module — exactly what FAISS and ScaNN do.  Those
optimization techniques work *there* because the per-iteration
cost is one memory access, not one interpreter round-trip.

## What this means for snapvec

I shipped what worked, archived what didn't with measured numbers
in a [`PERF_NOTES.md`](https://github.com/stffns/snapvec/blob/main/experiments/PERF_NOTES.md)
file in the repo, and changed the v0.5 release narrative from
"sub-millisecond at 1M" (impossible at this PQ rate without a
compiled kernel) to "ms-level latency at competitive recall, in
pure NumPy" (true and demonstrably so).

The v0.6 plan is to expose an opt-in `snapvec[fast]` extra that
adds a `numba` dependency and ships a kernel with the per-subspace
loop fused.  The base install stays single-`numpy` and the default
code paths keep their simplicity.  Users who actually need <1 ms
latency at 1M opt into the C-backed path.

## What you can take from this

If you're optimising NumPy code with a Python-level outer loop and
small per-iteration work, the moves that *will* pay off are:

- **Reduce iteration count**, not per-element bytes.  Restructure
  algorithms to do bigger chunks of work per call.
- **Match BLAS calling convention**: shapes that `gemv` / `gemm`
  can amortise win essentially for free.
- **Memory pressure**: if you're at the edge of RAM, *that* will
  dwarf any algorithmic win.  Profile peak memory before profiling
  CPU.

The moves that probably *won't* pay off, regardless of what the
folklore says:

- Multi-threading at the inner loop level.  If you use a BLAS-
  backed NumPy, you're already multi-threaded — adding workers on
  top means contention.
- Quantising arrays to smaller dtypes for memory bandwidth.  Only
  helps if the *bottleneck is bandwidth*; profile first.
- Algorithmic short-circuits.  Beautiful in theory; in pure NumPy
  the overhead of "checking whether to short-circuit" routinely
  costs more than the work it saves.

When all of those don't move the needle, the honest answer is to
drop into a compiled inner loop — and to be transparent with users
about which of your code paths require it.  Optional `numba`
extras are a clean way to do that without bleeding deps onto the
default install.

---

*All measurements are from a single MacBook Pro M-series, single
process, FIQA queries vs FIQA corpus from BeIR with BGE-small-en-v1.5
embeddings.  Reproduce with the scripts in `experiments/` on the
v0.5.0 release tag.*

*Comments / corrections welcome on the
[`snapvec` repo](https://github.com/stffns/snapvec/issues).*

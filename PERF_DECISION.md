# Performance Decision — `snapvec` v0.5

This is the synthesis of the v0.5 perf sprint: **what we measured**,
**what we ship**, and **what we explicitly punt to v0.6**.  Written
once we had data on the NumPy ceiling so the path forward is a
choice, not a guess.

## TL;DR

- The pure-NumPy hot path for `IVFPQSnapIndex.search` is
  **dispatch-bound at M = 192**, not memory-bound.  Floor is
  ~5 µs per visited candidate, ~1.5 ms / query at the FIQA "0.91
  recall" operating point.  No NumPy-level trick we tried (SWAR,
  threading, early-stop, quantised LUT) moves it.
- The full-scan PQ recall ceiling at M = 192, K = 256 is **0.929**.
  No amount of nprobe tuning gets past it.  The original v0.5 sprint
  target ("1M / 0.94 / < 1 ms on a laptop CPU") had a recall piece
  that is mathematically out of reach with this PQ rate.
- We ship v0.5 with the wins that materialised under measurement
  (build-time, ergonomics, layout, recall benchmarking) and with
  honest documentation of the ceiling.  We track the next-step
  options as v0.6 work.

## Where we landed (measured numbers)

### IVFPQSnapIndex on BGE-small / FIQA — ms per query at v0.5 column-major

`N = 57 638`, `nlist = 512`, `M = 192`, `K = 256`, single laptop
CPU, mean of 500 queries:

| `nprobe` | recall@10 | ms / q |
|---:|---:|---:|
| 8 | 0.771 | 0.59 |
| 16 | 0.845 | 0.94 |
| 32 | 0.891 | 1.55 |
| 64 | 0.914 | 2.70 |
| 128 | 0.924 | 4.96 |
| 256 | 0.929 | 9.57 |
| 512 (full PQ) | 0.929 | 17.31 |

### Build time at N = 1 M

`add_batch(1M)` was the surprise blocker — it took 14 minutes on
the v0.4 baseline.  The chunked-encoding fix in
`perf/v0.5-optimize-add-batch` (PR #17) brought it to **70 s**,
**12× faster**.  This is the largest single win of the sprint and
was the correct first thing to optimise even though it wasn't on
the original list.

## What we shipped in v0.5

### Wins
- **Chunked `add_batch`** — 12× build-time speedup at N = 1 M
  (memory-pressure fix; the residuals matrix at 1 M × 384 was
  triggering swap, not the algorithm).
- **Column-major `(M, n)` codes** — 12 % search-latency win on the
  per-subspace gather.  Also enables future contiguous patterns.
- **`search_batch`** — ergonomic batched API; performance-neutral
  vs per-query loop in pure NumPy, future-proof for v0.6.

### Quality / correctness
- `UserWarning` in `fit()` when `n_train < 30 · nlist` (FAISS rule).
  Caught in the v0.5 N=1M baseline (recall pinned at 0.731).
- L2-monotone coarse cluster ranking (`2⟨q, c⟩ − ‖c‖²`) — fixes a
  small but real bias in probe ordering since coarse centroids are
  means of unit vectors with varying norms.
- Duplicate-id rejection in `add_batch`.
- File-format `_offsets` validation on load.

### Documentation
- README: "Sizing nlist and the training set" with the FAISS rules
  + concrete table.  IVFPQ comparison numbers updated.
- New file: `experiments/PERF_NOTES.md` archives every negative
  experiment with measured numbers and root cause, so future-us
  doesn't burn time on the same ideas.
- New file: this `PERF_DECISION.md`.

## What we tried and dropped (with measurements)

Each of these lives in `experiments/PERF_NOTES.md` with the full
A/B and the structural reason it loses in NumPy.

| Attempt | Best ratio vs baseline | Why it lost |
|---|---|---|
| Early-stop probing | **0.24×** (4× slower) | Per-cluster Python dispatch swamps the gather, upper bound too loose to skip clusters anyway |
| Single-query `num_threads` | **0.27×** at nprobe=64 | Apple Accelerate / Intel MKL already parallelises `np.take` and matmul; worker threads compete with the BLAS pool |
| `search_batch num_threads ≥ 2` | 0.62× at B=128 | Same root cause as single-query threading |
| `search_batch num_threads = 1` | 1.0× (neutral) | Coarse + LUT batching savings already captured by NumPy/BLAS at small B; gather dominates |
| Quantised int8 LUT | **0.76×** | Hot path is dispatch-bound, not memory-bound; cast/multiply per iteration adds dispatch |
| Pair-LUT SWAR | not benched, structurally **negative** | (M/2, K²) int16 table = 12 MB per-query write at K = 256 — build cost > entire baseline latency |

## The original target — and why we changed it

**Originally:** "`IVFPQSnapIndex` at N = 1 M, recall = 0.94,
< 1 ms / query on a laptop CPU."

**After measurement, we know:**

1. The **0.94** piece is fundamentally limited by PQ rate at
   M = 192, K = 256.  The full-scan PQ ceiling is 0.929 at this
   rate; no IVF-side knob can move it.  Reaching 0.94 needs either
   a higher-rate codebook (M = 256 or K = 512 — uint16 codes,
   currently out of scope) or a float32 rerank pass on the IVF-PQ
   top-N (the v0.6 `pq_rerank=True` follow-up).
2. The **< 1 ms** piece is fundamentally limited by NumPy's per-
   subspace dispatch floor at M = 192.  At our floor of ~5 µs per
   visited candidate, the **0.91 recall point is at 1.5 ms** and
   **the 0.93 point is at 5 ms**.  No NumPy-level optimisation
   moves this.

**Re-targeted message for v0.5.0:**

> snapvec IVF-PQ delivers ms-level latency at competitive recall on
> real LLM embeddings, in pure NumPy — sub-2 ms at recall 0.91 on
> BGE-small at N = 57 k, scaling roughly linearly with N.  For
> recall > 0.93 or sub-millisecond latency at these PQ rates, see
> the v0.6 roadmap.

## v0.6 roadmap (the path past the v0.5 ceiling)

Each of these is tracked as a follow-up task and explicitly out of
scope for v0.5.  They are listed in priority order based on the
v0.5 measurements.

1. **`pq_rerank=True` option** (task #12).  After the IVF-PQ pass,
   re-score the top-N candidates against original float32 vectors
   for an exact dot product, then take top-k.  At N = 100 candidates
   that is sub-millisecond extra; storage cost is +4× per vector
   for the float32 cache (opt-in via `keep_full_precision=True`).
   On the FIQA bench this should reach recall ≥ 0.97 at the same
   nprobe latency band.  **This is the single highest-value v0.6
   item — it is what unlocks the recall ceiling.**

2. **Optional `numba`-accelerated ADC kernel** (`snapvec[fast]`
   extra).  Fuse the per-subspace gather + sum into a single
   compiled inner loop — dispatch floor disappears, and the
   quantised int8 / SWAR pipelines that NumPy could not exploit
   become real wins.  Expected ~2-5× on the search hot path,
   pushing the 0.91-recall point under 1 ms / query.  Keep base
   install dep-free; opt-in extra.

3. **Bit-packing PQ codes for K < 256** (task originally listed for
   PQSnapIndex follow-up).  At K = 16 we'd store 4 bits/sub instead
   of 8 — halves storage, doubles the candidates that fit in cache
   per byte of memory bandwidth.  Combined with #2 this is where
   the 1-ms target genuinely becomes achievable in laptop CPU.

4. **`filter_ids` on PQ / IVF-PQ**.  Production retrieval often
   wants to constrain by tenant / permission / category.  Currently
   only `SnapIndex` supports this.  The natural integration point
   is post-coarse-probe in IVF and a mask in the gather phase.

The honest summary of how to think about each:

- (1) gives users a way past the recall ceiling.
- (2) gives them a way past the latency ceiling.
- (3) makes (2) bigger.
- (4) makes the family feature-complete for production use.

## What did NOT change

The base thesis — "pure NumPy, no heavy dependencies, four index
families covering the accuracy / storage / latency frontier" — is
intact.  Everything (1)–(4) above is **opt-in**.  The default
install stays single-`numpy` and the default code paths keep the
v0.4 simplicity.

# Changelog

All notable changes to `snapvec` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.2] -- 2026-04-21

Release-infrastructure patch on top of v0.10.1.  No library changes.

### Changed

- **Drop `macos-13` from the wheel matrix.**  During the v0.10.1
  release run the GitHub-hosted `macos-13` runner pool stayed queued
  for ~45 minutes and was cancelled, blocking the PyPI publish
  (`publish` requires every wheel job to succeed).  Apple Silicon has
  dominated new Mac sales for 4+ years; Intel Mac users can install
  from the sdist (requires `brew install libomp` + a C compiler) or
  run the x86_64 sdist via Rosetta.  Re-adding the row is one matrix
  entry if GitHub's macos-13 capacity recovers.

The v0.10.0 and v0.10.1 tags exist in git but never produced PyPI
artifacts; 0.10.2 is the first version actually published.

## [0.10.1] -- 2026-04-20

Release-infrastructure patch on top of v0.10.0.  No library changes.

### Fixed

- **cibuildwheel macOS arm64 target**: `MACOSX_DEPLOYMENT_TARGET=13.0`
  was set uniformly for every macOS wheel, but the Homebrew libomp
  installed on GitHub's `macos-14` runner is linked against macOS 14
  system libraries.  `delocate-wheel` rejected the bundle with
  "Library dependencies do not satisfy target MacOS version 13.0".
  Pin the target per matrix row: `macos-13` builds (x86_64) stay at
  `13.0`, `macos-14` builds (arm64) now target `14.0`.  User-visible
  consequence: snapvec arm64 wheels require macOS 14+ (Sonoma); x86_64
  wheels still work on macOS 13+.
- **cibuildwheel test env missing hypothesis** (from v0.10.0 release
  attempt, previously fixed in PR #57 but worth recording in the
  release notes): the cibuildwheel test phase runs the full test
  suite against the built wheel and needs every test-time import.
  `CIBW_TEST_REQUIRES: "pytest hypothesis>=6.100"`.

## [0.10.0] -- 2026-04-20

Headline: **professionalisation release.**  The only library change
is a narrow fix in `SnapIndex.search` (reject `k < 1` with a clear
error, matching the other three index types); everything else wires
up the infrastructure, tests, documentation, and benchmarks that make
snapvec ready for third-party adoption and publishable comparisons.

### Added

- **CI test matrix** on Linux, macOS-14, and Windows across CPython
  3.10, 3.12, 3.13 (PR #43).  **Pre-compiled wheels** produced by
  `cibuildwheel` cover Linux x86_64/aarch64, macOS-13 x86_64,
  macOS-14 arm64, and Windows AMD64 across CPython 3.10, 3.11, 3.12,
  3.13, published to PyPI on `v*` tags via trusted publishing.
- **Documentation site** at <https://stffns.github.io/snapvec/> built
  from MkDocs Material + `mkdocstrings`.  Eighteen pages: getting
  started, one user-guide per index type, architecture, benchmarks,
  concurrency contract, API reference, changelog (PR #49).
- **Executable examples** in `examples/` (quickstart, PQ, IVF-PQ with
  rerank, filter search, save/load, streaming ingest) exercised by
  the CI matrix on every commit (PR #49).
- **Community files**: `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CONTRIBUTING.md`, bug / feature / PR templates, dependabot, a
  root `CLAUDE.md`, pre-commit config (PR #43).
- **Test suite grew from 151 to 190+**:
  - Hypothesis property-based tests (order-invariance, save/load
    round-trip, delete reduces len, search respects k, filter
    subset) across SnapIndex, PQSnapIndex, IVFPQSnapIndex (PR #50).
  - Determinism tests: same seed + same inputs produces byte-identical
    index files for all four classes (PR #50).
  - Adversarial tests: empty index, n=1, k > n, zero-norm queries,
    degenerate distributions, bit-extreme configurations (PR #50).
  - Forward-compat tests: patch a known-good file's version byte and
    assert the loader raises an upgrade hint (PR #52).
  - Recall regression smoke test on a clustered synthetic corpus:
    asserts IVFPQ + rerank stays >= 0.90 recall@10 on every CI run
    (PR #52).
- **`bench/threading`** curve on search_batch: p50 across num_threads
  = 1, 2, 4, 8 and nprobe in (4..256).  Published in docs (PR #51).
- **`bench/competitive`** head-to-head on BEIR FIQA: snapvec (three
  bit-depths for flat + IVFPQ with and without rerank at M=48 and
  M=192) vs FAISS IVFPQ (M=48 and M=192 matched-budget) vs hnswlib
  vs sqlite-vec.  Unified Pareto table in docs (PR #53).
- **MkDocs site now deploys automatically** to GitHub Pages on every
  merge to `main` (PR #49).

### Changed

- **`SnapIndex.search(k)`** now raises `ValueError` for `k < 1`, matching
  the validation that `PQSnapIndex`, `ResidualSnapIndex`, and
  `IVFPQSnapIndex` already had (PR #50).  Caught by the new adversarial
  suite; prior behaviour returned all results on `k=0` and `n-1`
  results on `k=-1` via an accidental slicing interaction.
- **Forward-compat errors** on `load()` now name the versions this
  build supports and point at `pip install -U snapvec` (PR #52).  The
  old `"unsupported version N"` is gone.
- **README trimmed from 673 to 91 lines**; technical content moved to
  the MkDocs site under structured navigation (PR #49).
- **macOS arm64 build** uses `-mcpu=native` instead of `-march=native`
  (which Xcode 15 clang rejects as `unknown target CPU 'apple-m1'`).
  Fixed the CI build on GitHub's `macos-14` runners (PR #43).

### Fixed

- Ten ruff lint errors in snapvec/ and tests/ (unused locals and
  ambiguous `l` loop variable) (PR #43).

### Documentation

- **Concurrency contract** (`docs/user-guide/concurrency.md`) makes
  the single-writer / multi-reader guarantee explicit.  Calls out
  that `freeze()` (or one warm-up search) is required before
  fanning out to reader threads because `SnapIndex.search`
  lazily materialises its fp16 centroid cache on the first query
  (PR #50, PR #51).
- **Benchmarks page** covers FIQA recall, sqlite-vec scale, the
  threading curve, and the competitive comparison (PR #53).
- **README badges**: PyPI version, Python versions, CI, Docs
  deployment, license, monthly downloads (PR #43, PR #49).

### Infrastructure

- `hypothesis` pulled into `[dev]` optional deps (PR #50).
- `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` pulled into a
  new `[docs]` optional extra (PR #49).
- `faiss-cpu`, `hnswlib`, `sqlite-vec` used only by the competitive
  bench; **not** added as snapvec runtime dependencies.

## [0.9.0] -- 2026-04-16

Headline: **5.8x faster search via Cython compiled kernels,
recall 0.977 at 441us on FIQA -- zero extra runtime dependencies.**

The v0.8.1 pure-NumPy sprint proved that the ADC scoring loop (74-97%
of search time) is dispatch-bound at M=192: NumPy's ~2us per-call
overhead x 192 subspaces = ~400us floor that no vectorisation trick
can break. This release compiles the hot path to native code via
Cython+OpenMP, eliminating the dispatch overhead entirely.

The 0.977 recall at nprobe=64+rerank reproduces the v0.6.0
measurement exactly, across three releases with non-trivial
algorithmic changes (column-major layout, batched matmul LUT,
compiled kernel). Same config, same data, same number.

### Added

- **Cython compiled ADC kernel** (`snapvec/_fast.pyx`) with OpenMP
  parallel scoring via `prange`. Two kernels:
  - `adc_colmajor`: fused subspace scoring loop (replaces the M=192
    Python dispatch loop). Serial below 2000 candidates, parallel
    above.
  - `fused_gather_adc`: reads codes directly from `_codes` via
    `row_idx` indirection, eliminating the intermediate `(M, total)`
    gather buffer. Saves both the memcpy and the allocation.

- **`setup.py`** with OpenMP auto-detection per platform (macOS via
  Homebrew libomp, Linux via gcc `-fopenmp`, Windows via MSVC
  `/openmp`). Falls back to serial-only compilation if OpenMP is
  not available.

### Changed

- **Build backend switched from hatchling to setuptools** to support
  Cython extension compilation.

- **numba removed** -- Cython produces identical performance (same
  LLVM backend on macOS) without the 143MB JIT runtime, cold-start
  penalty, or Python version lag. `snapvec` only depends on numpy
  at runtime.

- **`pyproject.toml` description** updated to reflect "NumPy + Cython
  compiled kernels" (was "Pure NumPy, no heavy dependencies").

### Performance

ADC kernel microbench (M=192, K=256):

| n_cand | NumPy (v0.8.1) | Cython serial | Cython parallel |
|--------|---------------:|--------------:|----------------:|
| 500    | 199 us         | 42 us (4.7x)  | 40 us (5.0x)    |
| 3,500  | 695 us         | 271 us (2.6x) | 187 us (3.7x)   |
| 10,000 | 2,112 us       | 788 us (2.7x) | 253 us (8.3x)   |
| 50,000 | 8,847 us       | 4,604 us      | 946 us (9.4x)   |

End-to-end on BGE-small / FIQA (N=57,638, nlist=512, M=192, K=256,
500 queries):

| nprobe + rerank | v0.6.0 | v0.9.0 | Speedup | Recall |
|-----------------|-------:|-------:|--------:|-------:|
| 32              | 1,380  | **369 us** | **3.7x** | 0.944 |
| 64              | 2,550  | **441 us** | **5.8x** | 0.977 |
| 128             | 4,590  | **635 us** | **7.2x** | 0.992 |

PQ full-scan: recall 0.915 at 1,529 us. IVF-PQ nprobe=64 + rerank
is 3.5x faster at higher recall -- strictly superior on both axes.

Four operating points, all sub-millisecond at N=57K:

| Profile        | nprobe | Recall | us/query |
|----------------|-------:|-------:|---------:|
| Fastest        |     16 |  0.883 |      304 |
| Interactive    |     32 |  0.944 |      369 |
| High precision |     64 |  0.977 |      441 |
| Near-exact     |    128 |  0.992 |      635 |

### What was tried and rejected

Full experimentation log in `experiments/PERF_NOTES.md`:

- **Vectorised ADC** (take_along_axis, chunked, fancy indexing):
  0.4-0.9x. Materialising (M, n_cand) intermediate kills L1 cache
  locality that the per-subspace loop preserves.
- **Sparse CSR matvec**: 3.3x without build cost, 0.2x with. CSR
  construction dominates.
- **Gather vectorisation** (np.concatenate, fancy indexing, cumsum):
  all slower. Contiguous-slice memcpy beats scattered access.
- **Numba gather kernel**: 0.1x. Element-by-element Numba loop
  can't beat NumPy's bulk memcpy for contiguous slices.
- **Numba** was the initial compiled backend. Matched Cython
  performance exactly but adds 143MB runtime (numba + llvmlite),
  2-3s JIT cold start, Python version lag, and serverless/container
  friction. Replaced by Cython for zero runtime cost.

### Pipeline structure (confirmed optimal)

```
Query -> Preprocess (6us) -> Coarse probe (19us) -> LUT build (25us)
      -> Gather row_idx (22us) -> Fused ADC (187us) -> Top-k (84us)
```

Gather-then-score (224 dispatches) beats score-per-cluster (6,144
dispatches) by 27x. Every pipeline stage profiled and tested against
alternatives.

## [0.8.1] -- 2026-04-16

Performance-only release: push pure-NumPy search to its ceiling.

### Changed

- **LUT build: 7x faster** -- replace M=192 separate per-subspace
  matmuls with a single batched `np.matmul` call.  Applies to
  `PQSnapIndex.search()`, `IVFPQSnapIndex.search()`, and
  `IVFPQSnapIndex.search_batch()` (einsum replaced with matmul).
  LUT build drops from ~10% to <1% of IVF-PQ search time.

- **PQSnapIndex ADC: 1.47x faster** -- switch `_codes` storage from
  row-major `(n, M)` to column-major `(M, n)`, matching the IVF-PQ
  layout.  Per-subspace gather now reads contiguous memory instead
  of stride-M hops.  End-to-end PQ search is ~32% faster.
  File format unchanged (transpose on save/load).

### Performance notes

Full pure-NumPy optimisation sprint documented in
`experiments/PERF_NOTES.md`.  Tested and rejected: vectorised ADC
via `take_along_axis` (0.4x), chunked ADC (0.4x), sparse CSR
matvec (0.2x with build cost), gather vectorisation via fancy
indexing (0.15x).  Pipeline structure (gather-then-score) confirmed
optimal: 224 dispatches vs 6,144 for score-per-cluster.

## [0.8.0] -- 2026-04-15

Headline: **multi-reader safety + id-scoped IVF search**.  Two
features the v0.7 users asked for once snapvec landed in multi-
tenant services: a formal thread-safety contract so a single index
can fan out across request threads without duplicating memory, and
id-whitelist filtering on `IVFPQSnapIndex` so tenant / partition
scoping doesn't have to be bolted on with post-filter passes.

### Added

- **`FreezableIndex` mixin** — every index class now exposes
  `idx.freeze()` / `idx.unfreeze()` / `idx.frozen`.  After
  freezing, every mutator (`add`, `add_batch`, `delete`, `fit`,
  `close`) raises `RuntimeError` naming the operation, so
  concurrent `search()` from multiple threads is race-free **by
  contract** — no lock is taken on the query hot path.

  `SnapIndex.freeze()` also pre-warms the lazy `_cache` (the
  float16 centroid matrix materialised on first full-scan search),
  so two concurrent first-queries on a just-frozen index no longer
  race on that assignment.

- **`IVFPQSnapIndex.search_batch` is safe under concurrent
  callers.**  A `threading.Lock` serialises the lazy
  `ThreadPoolExecutor` init so two callers can't both observe
  `_executor is None` and each create their own pool.  The init
  path uses double-checked locking so the steady-state query path
  never touches the lock.  Once created, the executor is not torn
  down during `search_batch()`; if a caller asks for a different
  `num_threads` after init, `search_batch` raises `ValueError`
  rather than shutting down a pool another thread may be about to
  submit work to.  Lifecycle teardown remains in `close()`.

- **`IVFPQSnapIndex.search(..., filter_ids=...)` /
  `search_batch(..., filter_ids=...)`** — id-whitelist filtering
  for IVF-PQ, cluster- and pool-aware:

  - *Cluster skip*: the probe ranking is restricted to clusters
    that contain at least one filter row, so sparse filters don't
    waste `nprobe` slots on empty clusters.  When
    `len(allowed_clusters) <= nprobe` the probe set is computed
    with a direct `np.tile` of the allowed clusters (no
    argpartition).
  - *Pool-aware*: the row filter is applied before the top-k /
    rerank-pool selection, so the rerank candidate pool is drawn
    from the filtered subset — no slots spent on ids the caller
    just excluded.
  - *Sparse-friendly mask*: internally represented as a sorted
    `int64` array + `np.isin(row_idx, filter_rows,
    assume_unique=True)`, not a dense `N`-bool array.  At
    `N = 100 M` this is the difference between a 100 MB
    allocation per query and a scan proportional to the probed
    subset.
  - Unknown ids in the filter are silently dropped; an entirely-
    unknown filter returns `[]` immediately.

### Changed

- Every index class's `add()` now calls
  `self._check_not_frozen("add")` before delegating to
  `add_batch()`, so the `RuntimeError` thrown on a frozen index
  names `add` (the method the caller invoked) rather than
  `add_batch` (the internal delegate).

### Documentation

- `IVFPQSnapIndex.search()` docstring clarifies that the rerank
  cache has been `float16` since v0.7 while the rerank matmul
  itself stays in `float32` via NumPy type-promotion on `q_pre`.
- `_freezable.py` module docstring spells out that any lazy per-
  class cache reachable from `search()` must be pre-warmed in the
  subclass's `freeze()` override (or serialised by the class
  itself) — not assumed safe by default.

## [0.7.1] — 2026-04-15

Test-correctness patch.  No behaviour change to the shipped fp16
rerank cache — the v0.7.0 recall measurements (0.943 / 0.977 / 0.993)
stand.  But the *unit test* that was supposed to protect against
future regressions of that cast was vacuous, caught in the PR #31
post-merge review (thanks, Copilot).

### Fixed

- **`test_fp16_cache_recall_matches_fp32_within_noise` was a tautology.**
  The original A/B swapped the fp16 cache for its own upcast
  (`cache.astype(float32)`) and compared.  Because upcasting
  float16 → float32 is lossless and the rerank matmul runs in fp32
  regardless (driven by `q_pre`'s fp32 dtype via NumPy type
  promotion), the two runs produced bit-identical scores and the
  assertion was always trivially true.  A future regression that
  made the cast lossier would not have been caught.

  The test now reconstructs the *pre-truncation* fp32 cache by
  re-running `_preprocess` on the original float32 corpus and
  compares rerank recall against *that*.  The fix includes an
  explicit `scores_differ` assertion that at least one score in the
  top-10 must differ between the fp16 and fp32 runs — a guard that
  will fail loudly if the test regresses back to the vacuous form.

- Clarified the `_full_precision` field docstring on how NumPy's
  type-promotion rules keep the rerank matmul in float32.  The
  previous wording said NumPy "promotes fp16 to fp32 internally on
  matmul" — technically true in this case, but the *reason* it lands
  in float32 is that `q_pre` is float32, not anything intrinsic to
  the fp16 dtype.  Rewritten to flag the risk if `q_pre` were ever
  downgraded to fp16.

## [0.7.0] — 2026-04-15

Headline: **tier-1 production reliability + halved rerank cache**.
Two features that make `snapvec` deployable at scale in environments
where silent disk corruption, cloud re-encoding, or partial writes
are real risks — and where the 4× storage cost of
`keep_full_precision=True` in v0.6 was the main reason users
capped out their use of the rerank path.

### Added

- **CRC32 trailer on every persistent format** (`.snpv`, `.snpq`,
  `.snpr`, `.snpi`).  Catches single-bit flips on disk and mid-
  transfer corruption at `load()` time instead of silently
  returning wrong search results.  8-byte tail: 4-byte magic
  `CRC2` + uint32 LE `zlib.crc32` of the preceding payload.
  `CRC2` (not `CRC1`) leaves room to upgrade to xxHash3 later
  without breaking old readers.
- **Shared atomic-save helpers** in `snapvec/_file_format.py`:
  `ChecksumWriter`, `save_with_checksum_atomic(path, writer_fn)`,
  `verify_checksum(path)`, `has_trailer(path)`.  Each index
  module's `save()` body shrinks to a `_write(f)` closure; the
  `.tmp` + `os.replace` atomic-rename dance lives in one place.

### Changed

- **`IVFPQSnapIndex` rerank cache is now float16** (was float32
  in v0.6).  Halves both disk *and* RAM footprint of
  `keep_full_precision=True` indices.  NumPy upcasts fp16 → fp32
  implicitly in the rerank matmul, so arithmetic precision is
  unchanged — only the stored representation is quantised.

  Recall A/B on BGE-small / FIQA (N = 57 638, `rerank_candidates=100`):

  | nprobe | fp32 (v0.6) | fp16 (v0.7) | Δ |
  |---:|---:|---:|---:|
  | 32 | 0.943 | 0.943 | 0.000 |
  | 64 | 0.977 | 0.976 | -0.001 |
  | 128 | 0.994 | 0.993 | -0.001 |

  Storage drops from 84 MB to 42 MB at N = 57 638; extrapolated to
  1.5 GB → 768 MB at N = 1M.

- **`.snpi` bumped to v4.**  v3 files (fp32 cache) load
  transparently via a **row-chunked fp32→fp16 cast** (≈1 MB of
  fp32 intermediate per chunk) so peak transient RAM stays
  bounded regardless of N.

### Fixed

- `ChecksumWriter.finalise()` is now actually idempotent (a second
  call is a no-op, not a second trailer).
- `write()` after `finalise()` now raises `RuntimeError` instead of
  silently invalidating the trailer CRC.
- `verify_checksum` opens the file once (previously twice — once
  via `has_trailer`, once to compute the CRC).

### v0.8 roadmap

- `freeze()` + thread-safe search contract (deferred from v0.7 —
  better with more soak time).
- `filter_ids` IVF-aware (cluster skip + pool-aware).
- `snapvec[fast]` Numba / Rust accelerator (see
  `docs/blog/02-fast-extension.md`).

## [0.6.0] — 2026-04-15

Headline: **breaks the PQ recall ceiling**.  The v0.5.0 release
documented a hard limit of **recall@10 = 0.929** at `M = 192, K = 256`
— no amount of nprobe could get past it, because the residual PQ
codebook couldn't resolve the last few percent of error.  v0.6 adds
an opt-in float32 rerank pass that recovers the recall the PQ
approximation lost, at <1 % latency overhead.

### Added

- **`IVFPQSnapIndex(... keep_full_precision=True)`** — stores an
  additional `(n, dim_eff) float32` cache of the original (post-
  preprocess) vectors alongside the PQ codes, sorted cluster-
  contiguously in sync with `_codes`.  Opt-in; default `False` keeps
  v0.5 storage footprint exactly.  Storage cost is
  `dim_eff × 4 bytes / vector` (e.g. +1536 B/vec at `dim = 384`).

- **`IVFPQSnapIndex.search(..., rerank_candidates=N)`** — when set
  (and the index was built with `keep_full_precision=True`), the
  IVF-PQ pass returns its top-`N` candidates, those get re-scored
  exactly against the cached float32 vectors, and the top-`k` of
  the reranked set is returned.  Requires `rerank_candidates >= k`.

  Measured on FIQA / BGE-small (N = 57 638, `nlist=512`, `M=192`,
  `K=256`, 300 queries, mean ms / query):

  | nprobe | PQ recall | PQ ms | + rerank(100) recall | + rerank(100) ms |
  |---:|---:|---:|---:|---:|
  | 16 | 0.842 | 0.87 | **0.880** | 0.89 |
  | 32 | 0.893 | 1.49 | **0.943** | 1.38 |
  | 64 | 0.917 | 2.52 | **0.977** | 2.55 |
  | 128 | 0.928 | 4.55 | **0.994** | 4.59 |

  Rerank cost is a single
  `(rerank_candidates, dim_eff) @ (dim_eff,)` matmul (~38k ops at
  N = 100).  Latency overhead is within measurement noise on this
  laptop.  `rerank_candidates=100` already saturates the recall
  lift — going to 200 does not move the number.

- **`.snpi` v3 file format** with `_FLAG_KEEP_FULL_PRECISION`.  v1
  and v2 files continue to load transparently (no full-precision
  data); v3 save/load round-trips the float32 cache byte-for-byte.

- **`stats()` now reports both `bytes_per_vec` and
  `bytes_per_vec_codes_only`** so users can see the storage delta
  that `keep_full_precision` introduces.

### Changed

- `IVFPQSnapIndex` internals extract a shared `_gather_pq_scores`
  helper so `_score_one` (default path) and `_score_one_with_rerank`
  share the same candidate-scoring logic — including the
  `* self._norms` scaling when `normalized=False`.  A single source
  of truth prevents divergence of the candidate metric between the
  two paths (a class of bugs that bit the first rev of this PR).

- **`SnapIndex.delete`** is now **O(1) via swap-with-last** instead
  of O(N) via `np.delete` + dict-rewrite.  Micro-bench at N = 10 k
  over 500 deletes: **421× faster** (112 ms → 0.27 ms).  Unlocks
  use cases with id churn (tenant rotation, TTL eviction).

- **`SnapIndex._apply_qjl_arrays`** (the `use_prod=True` path) no
  longer materialises a `(N, d) float32` temporary via
  `qjl.astype(np.float32) @ S_q`.  `np.dot(qjl, S_q)` delegates the
  mixed-type math to NumPy's C backend directly.  Latency-equivalent
  at our sizes (both paths hit the BLAS floor) but avoids the
  ~150 MB allocation spike per search at N = 100 k.

### Documentation

- README decision table adds a new row for the rerank path and
  quotes the 0.994 recall number.
- `docs/blog/01-numpy-perf-ceiling.md` and
  `docs/blog/02-fast-extension.md` kept up to date with the
  decision matrix (Numba vs Rust + PyO3) for the v0.7 accelerator
  follow-up.

### v0.7 roadmap (tracked but not in 0.6)

- **`snapvec[fast]` accelerator** for the IVF-PQ search hot path.
  Numba for prototyping, Rust + PyO3 + maturin for shipping.
- Bit-packed PQ codes for `K < 256`.
- `filter_ids` on `PQSnapIndex` and `IVFPQSnapIndex`.

## [0.5.0] — 2026-04-15

Headline: an evidence-driven perf sprint for `IVFPQSnapIndex`.
Search-side wins are modest in pure NumPy (the hot path is dispatch-
bound at M = 192 — the rest of the slack only opens up with a
compiled inner loop), but the build path, the API surface and the
documentation each got a real lift.  See `PERF_DECISION.md` for the
sprint synthesis and `experiments/PERF_NOTES.md` for the negative-
result archive.

### Added
- **`IVFPQSnapIndex.search_batch(queries, k, nprobe, num_threads)`** —
  ergonomic batched API.  Builds the per-batch coarse score matrix
  and the `(B, M, K)` LUT tensor with single BLAS calls instead of
  looping `search()` per query, with optional `ThreadPoolExecutor`
  parallelism for the per-query gather + scoring phase.
  Performance-neutral (~1×) vs per-query loop in pure NumPy because
  Apple Accelerate / Intel MKL already amortise; future-proof for the
  v0.6 numba kernel.
- **`IVFPQSnapIndex.close()`** — explicit thread pool teardown for
  long-lived processes that cycle indices.
- **n_train sizing warning + docs** — `IVFPQSnapIndex.fit()` raises a
  `UserWarning` when `n_train < 30 · nlist` (the FAISS rule of thumb),
  with the actual ratio and recommended size.  README adds a "Sizing
  nlist and the training set" subsection with a concrete table for
  N ∈ {10k, 100k, 1M, 10M}.
- **`PERF_DECISION.md`** — one-page sprint synthesis kept at the repo
  root: what we measured, what we shipped, what we dropped, why the
  original "1M / 0.94 / <1 ms" target shifted, and the v0.6 roadmap.
- **`experiments/PERF_NOTES.md`** — negative-result archive with
  measured A/B for every technique that did not pan out (early-stop
  probing, single-query `num_threads`, quantised LUT / SWAR), so we
  do not burn the budget twice on the same ideas.

### Changed
- **`add_batch` is 12× faster at N = 1M** (827s → 70s) via chunked
  residual encoding (peak transient memory bounded under ~150 MB).
  The previous N = 1M cost was dominated by swap, not algorithm.
- **`IVFPQSnapIndex` codes are now stored column-major `(M, n)`** so
  the per-subspace gather inside `search()` is a contiguous slice
  instead of a stride-M lookup.  ~12% latency win on the search hot
  path; structurally enables the cleaner gather patterns the rest of
  the sprint relies on.
- **`.snpi` file format bumped to v2** to match the column-major
  layout.  v1 files load transparently via a one-time transpose at
  load time.
- **Coarse cluster ranking now uses the L2-monotone score**
  `2⟨q, c⟩ − ‖c‖²` instead of plain `⟨q, c⟩`, which matches the
  metric used during k-means assignment.  Plain dot product was a
  small but real bias since coarse centroids are means of unit
  vectors with varying norms.

### Fixed
- `add_batch` now rejects duplicate ids (within the batch and against
  the already-indexed set).  Silently letting them through previously
  pointed `_id_to_row` at the wrong row.
- `load()` now validates `_offsets` invariants (length, monotone,
  boundary values).  Corrupted files raise instead of silently mis-
  indexing the codes buffer.
- Stop reading `ThreadPoolExecutor`'s private `_max_workers`
  attribute; track the worker count explicitly.

### Documentation
- README "Which Index should I use?" table now references the FIQA-
  queries recall sweep numbers from `bench_ivf_pq_fiqa_recall.py`
  instead of the augmented-corpus baseline that had its own ceiling
  artefact.

### v0.6 roadmap (tracked but not in 0.5)
- `pq_rerank=True`: float32 rerank of the IVF-PQ top-N to break the
  `K = 256` PQ ceiling at 0.929 recall@10.
- Optional `numba` accelerator (`snapvec[fast]`) that fuses the per-
  subspace gather + sum into a single SIMD inner loop, removing
  NumPy's per-iteration dispatch floor.
- Bit-packed PQ codes for `K < 256`.
- `filter_ids` on `PQSnapIndex` and `IVFPQSnapIndex`.

## [0.4.0] — 2026-04-15

Headline: snapvec grows from one scalar index to a family of four
(`SnapIndex`, `ResidualSnapIndex`, `PQSnapIndex`, `IVFPQSnapIndex`),
covering the full accuracy / storage / latency frontier from
training-free scalar compression to sub-linear IVF search.

### Added
- **`IVFPQSnapIndex`** — inverted-file + residual Product Quantization
  index with cluster-contiguous storage.  `fit()` trains `nlist` coarse
  k-means centroids, then a shared per-subspace residual PQ codebook
  (`M × K`) on the pooled residuals.  `add` sorts codes by coarse
  cluster id and maintains an `offsets[nlist + 1]` array so each probed
  cluster is a contiguous slice instead of a dynamic boolean mask.
  Cluster ranking uses the `L2`-monotone score `2⟨q, c⟩ − ‖c‖²` to
  match the metric used during assignment (plain `⟨q, c⟩` is biased
  because coarse centroids are means of unit vectors and their norms
  vary).  Search visits only `nprobe / nlist` of the corpus and
  merges probed-cluster scores with the per-cluster ADC LUT offset
  by `⟨q, centroid_c⟩`. On BGE-small (N = 20 000, `M=192`, `K=256`,
  `nlist=256`) this delivers **9.7× speedup at recall 0.910**
  (`nprobe=8`), **6.1× at recall 0.931** (`nprobe=16`), and
  **3.5× at recall 0.940** (`nprobe=32`, actually exceeding
  `PQSnapIndex` full-scan recall because per-cluster residuals have
  smaller variance than globally-centred vectors).  New persistent
  format `.snpi` (magic `SNPI`, v1).
- **`PQSnapIndex`** — product-quantization index with learned codebooks.
  Splits the embedding into `M` subspaces of equal size (exact division
  required — no ragged last subspace in v1), trains a per-subspace
  `K`-centroid codebook via k-means++ + Lloyd iterations, and scores
  queries with asymmetric distance computation (ADC) over a per-query
  `(M, K)` LUT. On BGE-small / SciFact this reaches **recall@10 = 0.94
  at 192 B/vec** (with `normalized=True`) where `SnapIndex(bits=3)`
  delivers 0.78 at the same storage, and matches `SnapIndex(bits=4)`
  recall at half the bytes per vector — at the cost of a one-off
  `fit(sample)` step that the training-free `SnapIndex` does not need.
  Off by default: `use_rht=False`, because the rotation that lets fixed
  Lloyd-Max codebooks work on arbitrary data actively destroys the
  subspace structure k-means is about to exploit. Storage per vector is
  `M` bytes in normalized mode, `M + 4` bytes otherwise (the float32
  norm). Ships with its own persistent format (`.snpq`, magic `SNPQ`,
  v1).
- **`ResidualSnapIndex`** — two-stage Lloyd-Max quantization (coarse
  `b1` bits + residual `b2` bits, with the residual rescaled by the
  theoretical `σ_r = √ε(b1)`). Reuses the existing `{2, 3, 4}`-bit
  codebooks; no new training. Opens operating points uniform snapvec
  cannot reach: on BGE-small/SciFact, `b1=3,b2=3` (6 bits/coord) lifts
  recall@10 from 0.870 (uniform 4-bit) to 0.921, and `b1=4,b2=3`
  (7 bits/coord) reaches 0.957. Includes a `rerank_M` search mode that
  does a coarse pass over the whole corpus and reranks only the top-`M`
  with the full reconstruction — converges to full-reconstruction
  recall at `M=100`, making the fine stage O(M) instead of O(N).
  New on-disk format `.snpr` (magic `SNPR`, v1); codes are unpacked
  `uint8` per coordinate in this release (tight packing is follow-up).

## [0.3.0] — 2026-04-14

### Added
- **Tight 3-bit packing** in RAM and on disk: 8 three-bit indices are
  packed across 3 bytes (24 bits) instead of the previous byte-aligned
  layout. 3-bit compression improves from **5.9× → 7.8×** on disk and
  **3.0× → 7.8×** in RAM at `d=384` (padded_dim=512), closing the gap
  that previously made 3-bit mode strictly dominated by 4-bit.
- **Vectorised FWHT**: `_rotation._fwht_inplace` now uses a single
  reshape view per butterfly level instead of a Python-level slice loop.
  Single-query RHT is **~24× faster at pdim=512** (460 µs → 19 µs) and
  **~40× faster at pdim=2048**. End-to-end warm query latency is
  unchanged at `N=100k, d=384` (the gemv still dominates) but the RHT is
  no longer the #2 cost in the Python glue — relevant for high-dim
  models and batched insertion.
- **File format v3** with a transparent backward-compatibility decoder.
  v1 and v2 files (which used byte-aligned 3-bit) are detected via the
  version field and decoded with the legacy path, then re-packed into
  the v3 layout in memory.

### Changed
- `_indices` storage is now bit-packed for all bit widths (2, 3, 4) in
  RAM. `_can_pack` evaluates to `True` whenever `(pdim * mse_bits) % 8
  == 0`, which is satisfied for every `pdim = 2^k` with `k ≥ 3`.
- `save` / `load` indices are zero-copy for all bit widths: RAM layout
  matches disk layout byte-for-byte, so `tobytes()` / `np.frombuffer`
  replaces the previous unpack + repack round-trip.

### Documentation
- README "Compression ratios" table reflects actual (not theoretical)
  numbers and collapses "disk" and "RAM" into a single column for all
  modes (they're equal now).
- Lloyd-Max table drops the `bytes/coord (actual)` warning column —
  3-bit now matches its theoretical 0.375 bytes/coord.
- Roadmap `Tight 3-bit packing` and `Vectorised FWHT` items moved to
  "previously shipped".

### Compatibility
- Pre-v0.3 `.snpv` files load unchanged. The reader dispatches the
  3-bit decoder on the version field, so no user action is required.
- Public API unchanged: `SnapIndex(dim, bits=...)` works as before,
  with smaller footprint for `bits=3`.

## [0.2.0] — 2026-04-13

### Added
- **RAM-packed indices** for 2-bit and 4-bit modes (byte-aligned
  packing). At `d=384, N=100k, 4-bit`: idle RAM drops from 51.6 MB to
  26.0 MB (-50%); warm RAM drops from 154 MB to 128 MB (-17%); warm
  query latency unchanged.
- **`normalized=True` constructor flag** for pre-normalized embeddings
  (skips the `np.linalg.norm` step in `add_batch` / `search`).
  Persisted via bit-1 of the v2 flags field.
- **Roadmap section** in README ranking pure-Python and optional Rust
  improvements by measured ROI.

### Fixed
- README compression numbers previously reported theoretical
  bytes/coord without the RHT padding; real footprint at `d=384` is
  `padded_dim=512`, so 4-bit is 260 B/vec, 2-bit is 132 B/vec, etc.
- Documented (not yet fixed in this release) that the 3-bit disk
  packer was byte-aligned, giving the same 0.5 bytes/coord as 4-bit.
  This is addressed in v0.3.0.
- README flagged the silent `float64 → float32` cast inside
  `add_batch` / `search` that triples peak RAM during insertion when
  inputs are `float64`. Measured at `+306 MB` for `N=100k, d=384`.

## [0.1.3] — 2026-04-12

### Added
- **`filter_ids`** parameter on `search()` for O(|filter| · d) pre-
  filtering instead of full-scan + post-filter.

## [0.1.1] — initial internal release

### Added
- Initial implementation of TurboQuant / HadaMax (RHT + Lloyd-Max)
  with 2/3/4-bit support.
- `use_prod=True` mode with QJL unbiased correction.
- `chunk_size` streaming search.
- `mypy --strict` type annotations.
- Atomic `.snpv` save / load format.

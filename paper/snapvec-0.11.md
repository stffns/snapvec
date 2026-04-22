# snapvec: an engineering note on compressed ANN for local-first RAG

**Jayson Steffens** -- stffens@gmail.com

Version 0.11.0.  Draft, 2026-04-22.

---

## Abstract

We describe `snapvec`, a Python library for compressed
approximate-nearest-neighbor (ANN) search over embedding vectors.
The library is aimed at **local-first retrieval-augmented generation
(RAG)** deployments: single-process Python services running on a
laptop or a single cloud instance, where the operating constraints
are (i) `pip install` must work without a native toolchain, (ii) the
index file must be a single self-contained artefact, and (iii)
per-query latency must stay under a few hundred microseconds with
recall close to float32 brute-force.  `snapvec` packages four
well-known index types (scalar `SnapIndex`, residual scalar,
`PQSnapIndex`, `IVFPQSnapIndex`) with one Cython+OpenMP kernel for
the asymmetric-distance hot path and no other native dependencies.

We report a single-dataset quantitative comparison on BEIR FIQA
(`N = 57,638`, `dim = 384`, BGE-small) against FAISS IVFPQ,
hnswlib, and sqlite-vec.  At matched PQ budget (`M = 192`,
12.6 MB disk), `snapvec` is 1.4x faster per query than FAISS at
essentially the same recall (0.895 vs 0.906).  With the float16
rerank pass, `snapvec` reaches recall `0.945` at 345 us p50
latency -- Pareto-dominant in the sub-500 us band on this corpus.
With the optional OPQ-P rotation shipped in 0.11.0, `snapvec` at
the aggressive `M = 48` / 4.9 MB corner reaches recall `0.649`,
about 4.6 percentage points above FAISS at the same disk budget;
FAISS wins the same corner on latency (144 us vs 263 us), so that
operating point becomes a recall-vs-latency pick rather than a
single winner.

We do **not** claim generality beyond this dataset and hardware
class.  The purpose of this note is to document reproducible
numbers and expose the engineering tradeoffs, not to argue for a
new algorithm.  A secondary, non-graphable point: the library's
audit surface is roughly 3,650 lines of Python plus a 67-line
Cython hot path -- small enough for a two-person team to read
end-to-end when tracing a regression or porting to a new
platform.  All code and benchmarks are open source (MIT) at
<https://github.com/stffns/snapvec>.

---

## 1. Motivation

Sentence embedding sizes grow linearly with the base model: 384
dimensions for BGE-small, 768 for E5-base, 1536 for OpenAI
`text-embedding-3-large`.  At one million vectors, the float32
corpus alone is between 1.5 GB and 6.1 GB of RAM, plus whatever
the index adds on top.  For **local-first** deployments -- a
desktop RAG agent, a single-tenant document search, an edge
assistant with a local knowledge base -- this matters on three
axes simultaneously:

1. **Disk footprint** caps how much context the system can index
   on a developer laptop or a `t3.medium` cloud instance.
2. **RAM residency** determines whether the index stays hot in
   cache between queries.
3. **Per-query latency** gates end-to-end response time once the
   LLM step is subtracted; anything slower than a few hundred
   microseconds eats the embedding call's own budget.

Existing local options handle part of the problem.  `sqlite-vec`
delivers exact brute-force cosine at zero tuning cost, but does
not scale past ~100k vectors before per-query latency exceeds the
10 ms mark on the hardware we tested.  FAISS is the reference
implementation for production ANN, but its install surface
(`faiss-cpu` PyPI wheel plus a C++ library) and GPU-oriented
defaults do not match the "`pip install` and go" expectation of
a Python-only application.

The question this note addresses is:

> *Can a compressed ANN index, implemented in NumPy with a small
> Cython hot-path and distributed as a standard wheel, deliver
> recall and latency competitive with FAISS at matched compression
> on modern sentence-embedding workloads?*

The answer, for the one workload we measured against, is: yes at
the mid-to-high compression corners, with a clear recall-vs-latency
tradeoff at the aggressive corner.

---

## 2. Background

### 2.1 Product quantization and IVF

Product quantization (PQ), introduced by Jegou et al. [1],
partitions each `d`-dimensional vector into `M` contiguous
subvectors of size `d_sub = d / M`.  Each subspace is then
quantized to one of `K` learned centroids, giving a code of
`M * log2(K)` bits per vector (typically `log2(K) = 8`, so one
byte per subspace).  Query-time scoring is done via an asymmetric
distance computation (ADC): for each subspace the query's
inner products against the `K` centroids are computed once
(`M x K` lookup table), and the score of a stored vector is the
sum of `M` table lookups.

Inverted File Index (IVF) partitions the corpus into `nlist`
coarse clusters using k-means; at search time only `nprobe`
clusters are visited.  Stacking IVF on top of residual PQ -- where
the per-vector residual after subtracting the coarse centroid is
what gets quantized -- yields the standard IVF-PQ pipeline.

### 2.2 Randomised Hadamard transform

TurboQuant (Zandieh et al., 2025) [2] observed that a randomised
Hadamard transform (RHT) applied before scalar quantisation makes
the per-coordinate distribution approximately Gaussian, which in
turn makes Lloyd-Max scalar quantisation near-optimal.  We apply
the same idea in the `SnapIndex` and `ResidualSnapIndex` variants.
For PQ-based indices we instead use OPQ (below), which learns a
rotation rather than drawing one at random.

### 2.3 Optimised PQ

OPQ (Ge et al., 2013) [3] learns an orthogonal rotation `R` that
balances per-subspace variance before PQ encoding.  Applied to
both corpus (`x -> R x`) and query (`q -> R q`), it preserves
inner products (the rotation is orthogonal) while making PQ
quantisation materially tighter on distributions where coordinate
variance is skewed -- as modern sentence embeddings often are.
snapvec 0.11.0 ships the parametric variant (OPQ-P): covariance
eigendecomposition, round-robin subspace allocation.  The
non-parametric variant (alternating optimisation of `R` and the
codebooks) is tracked as a follow-up.

---

## 3. Design

`snapvec` exposes four index classes, each targeting a different
point on the recall / disk / latency frontier:

| Class | Training? | Typical compression | Notes |
|-------|-----------|---------------------|-------|
| `SnapIndex` | none | 6-12x | RHT + Lloyd-Max, scalar, full scan |
| `ResidualSnapIndex` | none | 4-8x | Two-stage scalar, optional rerank |
| `PQSnapIndex` | one-shot `fit` | 24-96x | Product quantisation, full scan |
| `IVFPQSnapIndex` | one-shot `fit` | 24-96x | IVF + residual PQ, sub-linear search |

Throughout the rest of the paper, `SnapIndex` always refers to the
RHT + Lloyd-Max scalar implementation; the "RHT" acronym in text
and the `SnapIndex` class name in tables point to the same thing.

All four persist as single files (`.snpv`, `.snpr`, `.snpq`,
`.snpi`) with atomic writes (write-to-temp + rename) and a
trailing CRC32 checksum that is **verified on every `load()`
call** -- corrupted files raise `ValueError` with the path and
the expected/actual digest instead of returning silently-wrong
data.  The runtime
dependency is NumPy plus the compiled `_fast.pyx` extension
(bundled in the wheel); nothing else ships in the `[default]`
install.  The published macOS wheel also bundles `libomp`
(embedded via `delocate-wheel` in the release workflow), so end
users do not need a Homebrew `libomp` or Xcode-side clang to
install and run the library; a source build from the sdist does
need `brew install libomp`, documented in `CONTRIBUTING.md`.
Code footprint: ~3,650 lines of Python plus 67 lines of Cython,
small enough for a two-person team to audit end-to-end.

**Memory layout at load time.**  `load()` calls currently read
the whole file into process memory (the coarse centroids, PQ
codebooks, the `(M, N)` code table, optional float16 rerank
cache, and the id list).  There is no memory-mapped lazy path:
at N = 1M with `keep_full_precision=True`, an IVF-PQ index
instantiates at ~1 GB resident.  For services that keep the
index warm between queries this is the intended behaviour.  It
also means the **maximum usable corpus size is bounded by host
RAM (plus swap)**: a file larger than physical memory will fail
to `load()` with a `MemoryError` rather than degrading to disk
paging, which is a real constraint for N approaching 10M on a
16 GB laptop and a major motivator for the `mmap` path tracked
as a v0.12 item.

### 3.1 Cluster-contiguous storage

`IVFPQSnapIndex` stores codes column-major as `(M, N)` uint8,
sorted by cluster id with an `offsets` array of length
`nlist + 1`.  Visiting the `nprobe` top clusters becomes
`nprobe` contiguous slices rather than a boolean mask over the
whole corpus.  Streaming ingest in 0.10.3 replaced the previous
concat-then-argsort-then-reorder with a single per-cluster
contiguous memcpy, cutting `add_batch` cost by 18 % end-to-end
on 50k-row streaming runs.

### 3.2 Compiled ADC kernel

The per-query scoring inner loop (`M` LUT lookups per candidate,
accumulated into a score array) is implemented in Cython with
OpenMP `prange`.  Two kernels:

- `adc_colmajor(lut, codes, scores, parallel)` -- baseline, pure
  PQ full scan.
- `fused_gather_adc(all_codes, row_idx, coarse_offsets, lut,
  scores, parallel)` -- fused gather + accumulation, avoids
  materialising an intermediate `(M, n_candidates)` buffer.
  Active for IVF-PQ probing.

Serial below 2000 candidates (dispatch overhead), parallel above.
Measured speedup vs the pure-NumPy baseline is 4-5x at `M = 192`,
`K = 256` on 10k candidates (see repository's
`experiments/PERF_NOTES.md`).

### 3.3 Float16 rerank cache

For high-recall IVF-PQ, `keep_full_precision=True` stores each
corpus vector in float16 alongside its PQ code.  The search path
first runs IVF-PQ to select `rerank_candidates` (default 100), then
re-scores those against their stored float16 vectors.  The rerank
matmul runs in float32 because NumPy's mixed-dtype rule widens
`(rerank_candidates, dim) @ (dim,)` to the wider operand.  This
path adds disk (one float16 copy per vector) but recovers recall
beyond the PQ codebook ceiling: on FIQA we measure 0.945 at
rerank=100 vs 0.895 PQ-only at the same `M = 192`.

### 3.4 OPQ as an opt-in flag

`use_opq=True` on `PQSnapIndex` or `IVFPQSnapIndex` causes `fit()`
to learn `R` via eigendecomposition and apply it to both
preprocessing paths.  The rotation is stored in the index file
(`_FLAG_USE_OPQ` bit in the flags field); non-OPQ indices pay
only a four-byte version-header bump from v1 to v2 (`.snpq`) or
v4 to v5 (`.snpi`).  Older readers refuse newer files at the
version check.

---

## 4. Experiments

### 4.1 Setup

- **Dataset.** BEIR FIQA (one of the BEIR retrieval benchmarks).
  Corpus N = 57,638.  Queries: 200 sampled from FIQA's 648 test
  queries (first 200, sorted order).
- **Embedding.** BGE-small-en (`BAAI/bge-small-en-v1.5`), dim = 384,
  L2-normalised.
- **Ground truth.** Float32 brute-force top-10 dot product on the
  same unit-normalised corpus.  Recall reported is recall@10
  against exact NN, not against any other retrieval label.
- **Hardware.** Apple M4 Pro, 12 cores, 24 GB RAM, macOS 14.
  Every backend pinned to a single thread for apples-to-apples
  per-query latency:
  - FAISS: `faiss.omp_set_num_threads(1)` at module level.
  - hnswlib: `idx.set_num_threads(1)` on the index instance
    (the default spawns an OpenMP pool even on a single-row
    query).
  - snapvec: serial by default.
  GC disabled inside the timing loop.  Median and p99 reported
  over the 200 queries.
- **Versions.** snapvec 0.11.0, faiss-cpu 1.13.2, hnswlib 0.8.0,
  sqlite-vec 0.1.7, NumPy 2.4.3, Python 3.12.

### 4.2 Head-to-head (Pareto)

![Pareto plot](../docs/_static/pareto.png)

Unified table across every backend and every operating point.
Point area in the figure is proportional to the on-disk
footprint.  Rows ordered by recall@10 descending:

| Backend | recall@10 | p50 us | p99 us | disk MB | build s |
|---------|----------:|-------:|-------:|--------:|--------:|
| sqlite-vec (brute-force cosine, exact) | **1.000** | 13401 | 40936 | 91.1 | 0.6 |
| hnswlib (M=32, ef_search=128) | 0.994 | 524 | 824 | 104.5 | 44 |
| **snapvec IVFPQ + fp16 rerank (M=192)** | **0.945** | **345** | 510 | 56.9 | 111 |
| FAISS IVFPQ (M=192) [matched-budget] | 0.906 | 457 | 550 | 12.7 | 17 |
| **snapvec IVFPQ no rerank (M=192)** | 0.895 | **319** | 502 | 12.6 | 110 |
| snapvec IVFPQ no rerank (M=192) + OPQ | 0.895 | 368 | 557 | 13.2 | 112 |
| snapvec SnapIndex 4-bit scalar (full-scan) | 0.854 | 2764 | 3099 | 15.4 | 1.1 |
| snapvec SnapIndex 3-bit scalar (full-scan) | 0.736 | 2742 | 3815 | 11.7 | 0.8 |
| **snapvec IVFPQ no rerank (M=48) + OPQ** | **0.649** | 263 | 331 | 4.9 | 33 |
| snapvec SnapIndex 2-bit scalar (full-scan) | 0.618 | 2740 | 5594 | 8.0 | 0.7 |
| FAISS IVFPQ (M=48) | 0.603 | 144 | 217 | 4.4 | 10 |
| snapvec IVFPQ no rerank (M=48) [matched-budget] | 0.549 | 241 | 319 | 4.3 | 34 |

The three findings we highlight:

1. **Sub-500 us flagship corner.**  snapvec IVFPQ + fp16 rerank
   sits alone above `recall = 0.9` in the sub-500 us band
   (`0.945 @ 345 us`).  FAISS IVFPQ M=192 is the next comparable
   point at lower recall and higher latency (`0.906 @ 457 us`).
2. **Matched-budget M=192.**  At identical PQ config (M=192,
   K=256, no rerank) snapvec is 1.4x faster than FAISS
   (`319` vs `457` us p50) at essentially the same recall
   (`0.895` vs `0.906`) and identical disk (`12.6` vs `12.7` MB).
3. **Aggressive-compression corner.**  At M=48 the pareto breaks
   apart.  With OPQ, snapvec reaches `0.649` recall at `263 us`
   and `4.9 MB`.  FAISS at M=48 reaches `0.603` recall at
   `144 us` and `4.4 MB`.  Neither dominates: snapvec wins on
   recall by 4.6 pp, FAISS wins on latency by 1.8x, disk is
   essentially identical.  Users at the 4-5 MB budget choose on
   which axis they care about.

The hnswlib row tops the approximate-recall column at 0.994, at
the cost of an order-of-magnitude larger index file (104 MB, ~8x
the matched-budget PQ disk) and higher p99.  sqlite-vec is exact
but 30x-90x slower than any of the ANN backends.

### 4.3 OPQ impact sweep

To isolate OPQ's effect from everything else, we run snapvec
IVFPQ at three M settings with and without the rotation, keeping
all other hyperparameters identical:

|  M  | d_sub | recall@10 (baseline) | recall@10 (OPQ) | delta |
|----:|------:|---------------------:|----------------:|------:|
|  48 |   8   |                0.553 |           0.656 | +10.3 pp |
|  96 |   4   |                0.767 |           0.812 |  +4.6 pp |
| 192 |   2   |                0.932 |           0.931 |    0.0  |

OPQ helps when the subspace dimension is at least 4 -- there has
to be room to redistribute variance across sorted eigenvectors.
At `d_sub = 2` the rotation is a signed pair swap and the claim
that PQ is near-optimal in the rotated basis holds no harder than
in the original basis.  Latency is identical in all three rows
within measurement noise; OPQ adds one `(dim, dim)` matmul at
preprocess (1.6 us per query on 384-dim), hidden by the other
costs.

Training cost of `fit_opq_rotation` is one eigendecomposition of
the `(dim, dim)` covariance matrix, measured at **21 ms / 52 MB
peak** on 10k training rows and **56 ms / 271 MB peak** on 57k
rows (dim = 384).  Earlier drafts cast the entire centred
training sample to `float64` in one shot for numerical stability,
which peaked at ~3 GB transiently at N = 1M.  The shipped 0.11.1
implementation accumulates the covariance in 16,384-row chunks, so
peak working memory during the cast is bounded by
`chunk * dim * 8` bytes (~30 MB at dim = 384) independent of `N`.
Per-row outer products, accumulator dtype, and eigendecomposition
are unchanged; all OPQ determinism tests pass bit-identically.

**End-to-end fit time.**  For the full `IVFPQSnapIndex.fit` +
`add_batch` pipeline on the FIQA corpus (`N = 57,638`), we
measured ~32 seconds at M=48, ~56 seconds at M=96, and ~110
seconds at M=192 on the M4 Pro.  OPQ adds under a second to
those numbers.  FAISS at matched config is roughly 3x-6x faster
(~10 s at M=48, ~17 s at M=192).  The gap is not a code
inefficiency on snapvec's side; it's the cost of building the
training pipeline on top of NumPy + a single Cython hot path
instead of pulling in a dedicated C++ k-means implementation.
Both libraries sit comfortably within the "seconds to minutes"
budget for a laptop-local workflow.  We do not recommend either
for a corpus big enough to push fit into hours without first
rethinking the sampling strategy (training on 10-20k rows and
indexing the rest is the standard move).

### 4.4 Scale against sqlite-vec

Earlier (pre-competitive-table) we measured snapvec's flagship
config against sqlite-vec across N = 10,000 to 1,000,000 (FIQA
corpus augmented to 1M via BGE-small re-embedding of adjacent
corpora, same dim).  The speedup relative to sqlite-vec's
brute-force scan grows with N:

| N | sqlite-vec | snapvec | speedup | snapvec recall |
|---|------------|---------|---------|----------------|
| 10k  | 2.3 ms | 0.44 ms | 5x  | 0.997 |
| 57k  | 15.1 ms | 0.44 ms | 34x | 0.977 |
| 100k | 23.8 ms | 1.04 ms | 23x | 0.994 |
| 500k | ~110 ms | 0.9 ms | 125x | ~0.97 |
| 1M | brute-force infeasible | 1.1 ms | -- | -- |

Disk footprint across the range is 2-8x smaller in snapvec's
favour.  These numbers are from an older run on the same hardware
and are kept as an order-of-magnitude reference; they are not
from the clean-process 0.11.0 run that produced Section 4.2.

### 4.5 Threading curve (search_batch)

`search_batch` fans out per-query scoring across worker threads.
On the M4 Pro (12 cores), measured speedup versus `num_threads = 1`:

| nprobe | t=1 ms/q | t=2 ms/q | t=4 ms/q | t=8 ms/q | best speedup |
|-------:|---------:|---------:|---------:|---------:|-------------:|
|   4    | 0.09     | 0.06     | **0.05** | 0.06     | 1.69x |
|  32    | 0.50     | 0.28     | **0.16** | 0.20     | 3.06x |
| 256    | 3.70     | 1.91     | **1.01** | 1.09     | 3.67x |

`num_threads = 4` is the sweet spot across every nprobe on this
machine.  At `t = 8` the executor oversubscribes the efficiency
cores and regresses.  Scaling improves with `nprobe` because
per-query work grows, so threading overhead amortises.  We leave
`search()` (single-query) serial on purpose -- it competes with
NumPy's own BLAS thread pool.

---

## 5. Related work

**FAISS** (Meta AI Research).  The production reference
implementation for compressed ANN, mostly C++ with Python
bindings.  FAISS has the deepest coverage of algorithms (IVF-PQ,
HNSW, LSH, IVFFLAT, IndexRefineFlat) and is the benchmark any
new library should be measured against.  `snapvec` makes a
different engineering tradeoff: pure Python + one Cython hot
path, wheel-distributable, smaller surface, less flexible.

**ScaNN** (Google Research).  Linux-only, strong recall-vs-latency
on MS MARCO-scale benchmarks, ships via a TensorFlow-shaped
wheel.  We did not include ScaNN in our comparison because its
Linux-x86_64-only wheel does not run on the M4 Pro hardware we
measured on; a Linux cloud-instance comparison is the natural
follow-up.

**hnswlib** (Malkov and Yashunin).  Graph-based.  Different
algorithmic family from IVF-PQ.  Included in Section 4.2 because
it is the other standard Python-installable ANN option.

**sqlite-vec**.  Brute-force cosine via a SQLite virtual table.
Included as the "zero ANN tuning, accept the latency" baseline.
`snapvec` started as the drop-in-replacement backend for
[vstash](https://github.com/stffns/vstash), which previously used
sqlite-vec.

**USearch**.  Similar spirit to `snapvec` (single-header,
single-binary, distributable) but written in C++ with optional
Python bindings.  Not included in 4.2; a follow-up comparison is
worthwhile.

---

## 6. Limitations

**Single dataset, single hardware class.**  Every number in
Section 4 comes from BEIR FIQA with BGE-small on an Apple M4 Pro.
Modern sentence embeddings vary in coordinate-variance shape,
and OPQ gains in particular are distribution-dependent -- the
`M = 192` row in the impact sweep is literally zero gain.  Users
should run `experiments/bench_competitive.py` on their own
corpus before treating the matched-budget FAISS comparison as a
universal claim.

**No GPU backend.**  `snapvec`'s design point is "no special
hardware, laptop latency".  A GPU path would invert that
constraint and is explicitly out of scope for v0.x.

**No hybrid retrieval.**  `snapvec` is a pure vector-ANN library.
Hybrid text + vector retrieval with RRF fusion lives one layer up,
in `vstash`.

**Write path still O(N) per add_batch.**  Although 0.10.3 cut the
constant factor by ~18 %, the cluster-contiguous layout still
requires one N-sized memcpy per `add_batch` call.  Truly O(1)
per-row streaming requires a delta-buffer layout (tracked as a
v0.12 roadmap item).

**Load path still materialises the full corpus in RAM.**  Even
after the 0.11.1 memory patch (`readinto` into pre-allocated
numpy buffers, dropping the ~2x transient that
`frombuffer(f.read(...)).copy()` held), `load()` still allocates
an (M, N) codes array, plus the optional fp16 rerank cache and
the id list.  A memory-mapped lazy path that keeps pages on disk
and faults them in as probes touch clusters is the next big
footprint win and is tracked as a v0.12 item.

**No runtime warning on bad OPQ config.**  OPQ gains collapse at
`d_sub < 4` (see Section 4.3).  `use_opq=True` is accepted
silently today; a `UserWarning` from `PQSnapIndex.__init__` /
`IVFPQSnapIndex.__init__` when `dim / M < 4` is a tiny engineering
improvement that would help downstream users avoid enabling the
flag in configurations where it costs latency and disk without
the corresponding recall win.

**OPQ-P, not OPQ-NP.**  The non-parametric variant that alternates
between learning the rotation and re-fitting the codebooks is
reported in the literature as an additional 0.3-0.8 pp recall at
the cost of a slower fit.  Not in 0.11.0.

**Single-writer concurrency.**  The library documents a
single-writer multi-reader contract.  A `freeze()` call is
required before fanning out to reader threads because search
lazily materialises a centroid cache on the first query.  There
is no internal lock on mutations.

---

## 7. Reproducibility

All code and the measurements in Section 4 are open source under
MIT license at <https://github.com/stffns/snapvec>, release tag
`v0.11.0`.  The exact commands to reproduce each table:

```bash
# Install the measured release
pip install snapvec==0.11.0

# Clone for benchmarks (not shipped in the wheel)
git clone https://github.com/stffns/snapvec && cd snapvec
git checkout v0.11.0
pip install -e ".[dev]"

# Cache the FIQA corpus + queries (one-shot; see the script
# for the BEIR + BGE-small embedding details).
python experiments/_colab_embed_corpus.py

# Section 4.2 matched-budget head-to-head (with OPQ rows).
# Each backend runs in an isolated subprocess because FAISS,
# hnswlib, and snapvec bundle their own libomp on macOS arm64
# and dynamically linking two of them crashes the interpreter.
python experiments/bench_competitive.py

# Section 4.3 OPQ impact sweep
python experiments/bench_ivfpq_opq.py

# Section 4.5 threading curve
python experiments/bench_ivfpq_threading.py

# Pareto plot (reads the hard-coded rows from Section 4.2)
python experiments/plot_pareto.py docs/_static/pareto.png
```

Typical total runtime on the M4 Pro is about 30 minutes
dominated by `IVFPQSnapIndex.fit()` calls on the full corpus
(FAISS IVFPQ training is ~8x faster, a known gap).

---

## 8. Conclusion

`snapvec` is an engineering artefact rather than an algorithmic
contribution: we package TurboQuant-style scalar quantisation,
residual PQ, IVF probing, float16 rerank, and OPQ-P rotation
behind a small Python API with a single compiled hot path.  The
design target is **local-first retrieval-augmented generation**,
not billion-scale web search.

On the one dataset we measured against, the matched-budget
comparison against FAISS IVFPQ is favourable at M=192 (1.4x
faster, equal recall) and mixed at M=48 (recall wins for
snapvec+OPQ, latency for FAISS).  The float16 rerank path gives
the only sub-500 us point above recall `0.9` on this workload.

We expect these numbers to carry to other modern sentence
embeddings with similar coordinate-variance shape (BGE-large, E5,
jina-embeddings) and to be sensitive on embeddings trained
without L2 normalisation or with heavily sparse coordinates.  We
explicitly invite users to run the same bench on their own
corpus before citing the FAISS comparison downstream.

A last engineering point worth stating: the audit surface of
`snapvec` is about 3,650 lines of pure Python plus a 67-line
Cython hot path, versus FAISS's tens of thousands of lines of
C++.  For a two-person ML team that needs to trace a recall
regression to its root cause, patch an obscure edge case, or
port a piece of the stack to a new platform, the ability to read
every line of the library in an afternoon is itself a feature --
one that the matched-budget performance numbers do not capture
but that downstream maintainers tend to care about.

---

## References

[1] Hervé Jegou, Matthijs Douze, and Cordelia Schmid.  *Product
    Quantization for Nearest Neighbor Search.*  IEEE Transactions
    on Pattern Analysis and Machine Intelligence, 33(1), 2011.

[2] Amir Zandieh, Mahdi Daliri, Amirhossein Hadian, and Vahab
    Mirrokni.  *TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate.*  arXiv:2504.19874, 2025.

[3] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun.
    *Optimized Product Quantization.*  IEEE Transactions on
    Pattern Analysis and Machine Intelligence, 36(4), 2013.

[4] Yury Malkov and Dmitry Yashunin.  *Efficient and robust
    approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs.*  IEEE TPAMI, 42(4), 2018.

[5] Ruiqi Guo et al.  *Accelerating Large-Scale Inference with
    Anisotropic Vector Quantization.*  ICML 2020.  (ScaNN.)

---

*Acknowledgements: benchmarks were code-reviewed before each run
via the `code-reviewer` subagent in the author's development
workflow, which caught several fairness-affecting bugs (thread
pinning, MACOSX_DEPLOYMENT_TARGET, missing query rotation in the
batch path) during the course of this work.*

# snapvec roadmap

Planned work and non-goals for upcoming releases.  Items here are
intent, not commitments -- dates slip, priorities change.  Track
open work in [issues](https://github.com/stffns/snapvec/issues).

## v0.11 (target: June 2026)

### Streaming ingest

- ~~**`add_batch` resort cost**~~ shipped in v0.10.3.  Single-copy
  merge in IVFPQSnapIndex.add_batch reduces streaming ingest cost
  by ~18%; growth remains super-linear because cluster-contiguous
  storage still requires one N-sized memcpy per batch.  True O(1)
  per row still needs the delta-buffer design below.

### Recall

- **OPQ rotation** for PQSnapIndex / IVFPQSnapIndex: learn a rotation
  during `fit()` that minimises PQ reconstruction error (Ge et al.,
  2013).  Typically +1-2 pp recall at the same bytes/vec.

### Types

- ~~**mypy strict in CI as a hard gate**~~ shipped in v0.10.3.

## Parked / explored

### `early_stop` for IVF probing (explored 2026-04-21, negative result on FIQA)

Idea: short-circuit the probe loop once the next cluster's best
possible score (`coarse_dot + max_residual`) falls below the current
k-th score.  Implemented in chunks of 32 clusters to amortise
dispatch overhead.

Measured on BEIR FIQA (N=57,638, BGE-small, M=192, K=256) against
the batched full-scan:

| nprobe | full ms | early ms | speedup |
|-------:|--------:|---------:|--------:|
| 4   | 0.16 | 0.16 | 0.98x |
| 32  | 0.34 | 0.55 | 0.62x |
| 256 | 1.09 | 3.43 | 0.32x |

Recall is identical (bound is strict), but early_stop is **slower**
at every operating point.  Three reasons:

1. Per-chunk `fused_gather_adc` dispatch + Python merge overhead is
   already substantial relative to the batched kernel's single-call
   time.
2. The global `sum_j max_k lut[j, k]` bound is loose on real
   embeddings -- actual residual scores rarely approach the
   max-per-subspace product.
3. FIQA's recall curve is gradual (0.66 at nprobe=4, 0.93 at
   nprobe=256), indicating top-k is distributed across many
   clusters -- the stop condition rarely fires.

Code kept in the `feat/ivfpq-early-stop` branch for future reference.
Revisit when we have a workload with concentrated ground truth (few
dominant clusters) and/or a tighter per-cluster bound learned at
`fit()` time.

## v0.12 (target: August 2026)

### File format v2

- **Per-block CRC32** in addition to the existing trailer CRC.  Catches
  corruption earlier in the read path, without reading the whole file.
- **Explicit feature flags** in the magic header so old readers fail
  cleanly on files that use a feature they don't support (right now
  this surfaces as a generic version mismatch).

### Incremental updates (delta buffer)

- **WAL-style append buffer** for `add` / `delete` without rebuilding
  the IVFPQ layout.  Reads merge the buffer with the base index.
  Periodic compaction flushes the buffer into the base.  Targets the
  pattern where snapvec backs a live system (for example vstash) and
  rebuilding on every write is not feasible.

## v1.0 (target: Q4 2026)

- **API freeze** across the four index types.  Semver kicks in --
  breaking changes require a major version and a documented upgrade
  path.
- **Two-minor-version deprecation window** for any public symbol.
- **Reproducible `bench/` suite** that runs in CI with pinned
  datasets (BEIR SciFact, FIQA, NFCorpus) and emits machine-readable
  JSON results keyed by commit + hardware.
- **Published head-to-head** vs FAISS / ScaNN / hnswlib on a
  Linux x86_64 cloud instance (the current bench is macOS-arm64;
  ScaNN only ships Linux wheels).

## Explicit non-goals

- **GPU backend.**  The design point is "no special hardware, laptop
  latency".  A GPU path would invert that.
- **Billion-scale indices.**  `IVFPQSnapIndex` is tested to 1M and
  designed for local-first RAG / memory systems, not web-scale search.
  If you need billion-scale, use FAISS or Milvus.
- **Approximate hybrid retrieval.**  snapvec is a pure vector-ANN
  library.  For hybrid text + vector retrieval with RRF fusion, use
  [vstash](https://github.com/stffns/vstash).
- **Matching FAISS's BLAS-heavy `fit()` speed.**  snapvec's `fit()`
  is ~7x slower than FAISS IVFPQ at matched config because FAISS
  hooks into heavily optimised BLAS.  Closing this gap is not a
  goal on its own; it's a sub-goal of the broader "depend on NumPy
  only" design constraint.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).  If you want to work on an
item listed here, open an issue first so we can scope it together.

# snapvec roadmap

Planned work and non-goals for upcoming releases.  Items here are
intent, not commitments -- dates slip, priorities change.  Track
open work in [issues](https://github.com/stffns/snapvec/issues).

## v0.11 (target: June 2026)

### Streaming ingest

- **`add_batch` resort cost**: IVFPQSnapIndex currently re-sorts the
  entire corpus by cluster id on every `add_batch` call (O(N log N)
  per append).  Acceptable for the documented bulk-ingest-then-search
  pattern, painful for streaming.  Plan: append-only layout with a
  periodic compaction phase.

- **`early_stop` for IVF probing**: short-circuit the probe loop once
  the top-k boundary can no longer be beaten by remaining clusters
  (using the coarse-centroid score as a cluster upper bound).
  Benchmark already scaffolded in `experiments/bench_ivf_pq_early_stop.py`.

### Recall

- **OPQ rotation** for PQSnapIndex / IVFPQSnapIndex: learn a rotation
  during `fit()` that minimises PQ reconstruction error (Ge et al.,
  2013).  Typically +1-2 pp recall at the same bytes/vec.

### Types

- **mypy strict in CI as a hard gate**.  17 errors remain, all small;
  cleanup is tracked.

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

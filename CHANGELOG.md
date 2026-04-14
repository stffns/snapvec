# Changelog

All notable changes to `snapvec` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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

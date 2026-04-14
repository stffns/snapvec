# snapvec

**Fast compressed approximate nearest-neighbor search.  Pure NumPy.  No heavy dependencies.**

`snapvec` implements the TurboQuant compression pipeline — randomized Hadamard transform followed by optimal Gaussian scalar quantization (Lloyd-Max) — as a self-contained Python library for embedding vector search.  It achieves **~6× / ~8× / ~12× compression at 4-bit / 3-bit / 2-bit** (disk and RAM match, thanks to tight bit-packing) with **>0.92 recall@10** against float32 brute-force, using only NumPy.

```
pip install snapvec
```

### Which `bits` should I use?

| If you are… | Pick | Why |
|-------------|------|-----|
| Building RAG / semantic search (the default) | **`bits=4`** | ~95% recall@10 on real embeddings, 5.9× smaller than float32 |
| Squeezing massive corpora where scale beats precision | **`bits=2`** | 11.6× smaller; recall@10 ≈ 0.83 on synthetic, higher on clustered real data |
| Willing to pay a bit of accuracy for ~25% more compression vs 4-bit | **`bits=3`** | 7.8× smaller, now tightly packed — a genuine middle ground as of v0.3.0 |
| Need unbiased inner-product estimates (KV-cache, attention) | **`bits=3` or `4` + `use_prod=True`** | QJL correction at the cost of ~2× search latency |

### `SnapIndex` vs `PQSnapIndex`: training-free vs trained

`snapvec` ships two index types with different trade-offs:

| Use | Index | Training | Typical gain |
|-----|-------|----------|--------------|
| Drop-in, no offline step, stable across datasets | **`SnapIndex`** (RHT + fixed Lloyd-Max codebooks) | none | baseline |
| You can spend a few seconds training codebooks on a sample of your corpus | **`PQSnapIndex`** (product quantization, k-means codebooks) | one-off `fit(sample)` | +10–18 pp recall@10 at matched bytes/vec on real embeddings |

On BGE-small / SciFact, `PQSnapIndex(M=192, K=256)` reaches recall@10 = 0.94 at 192 B/vec — `SnapIndex(bits=3)` delivers 0.78 at the same storage; `PQSnapIndex(M=128, K=256)` matches `SnapIndex(bits=4)` at half the bytes per vector. See `experiments/bench_pq_scaleup_validation.py` for the full sweep (3 seeds, K ∈ {16, 64, 256}, disjoint train/eval split).

---

## Quick start

```python
import numpy as np
from snapvec import SnapIndex

# Build index
idx = SnapIndex(dim=384, bits=4)          # 4-bit, ~6x compression
# ⚠ pass float32 arrays — see "Input dtype" below
embeddings = np.asarray(embeddings, dtype=np.float32)
idx.add_batch(ids=list(range(N)), vectors=embeddings)

# Query
results = idx.search(query_vector.astype(np.float32), k=10)  # [(id, score), ...]

# Persist
idx.save("my_index.snpv")
idx2 = SnapIndex.load("my_index.snpv")   # atomic save, v1/v2/v3 compatible
```

### Input dtype: pass `np.float32`

`add_batch`, `add`, and `search` accept any array-like input but cast
internally to `float32`. **If you pass `float64` (the NumPy default),
a full-size temporary copy is allocated during the cast** — for
`add_batch(1M × 384)` that's a transient 1.5 GB allocation on top of
your input array, gone only after quantization completes.

To avoid this:

```python
# Models that return float32 directly (most modern embeddings)
embeddings = model.encode(texts)             # already float32 → no copy
idx.add_batch(ids, embeddings)

# Arrays from np.array([...]) default to float64 — cast explicitly
embeddings = np.asarray(my_list, dtype=np.float32)   # cast once upfront
idx.add_batch(ids, embeddings)

# Loading from disk — respect the original dtype or force float32
embeddings = np.load("vecs.npy").astype(np.float32, copy=False)
```

The cast is a correctness convenience, not a design choice — the whole
pipeline (normalize, RHT, quantize) operates in `float32`.

---

## Technical background

### The problem: embedding vectors are expensive

Modern embedding models produce float32 vectors of dimension `d ∈ {384, 768, 1536}`.
Storing N vectors requires `4·N·d` bytes; brute-force search costs `O(N·d)` per query.
For N = 1M, d = 384: **1.5 GB RAM**, with inner products dominating inference time.

Product Quantization (PQ) splits vectors into M sub-vectors and quantizes each
independently. It is effective but requires training a K-means codebook per dataset.
Random Binary Quantization (RaBitQ, 1-bit) is fast but coarse.

**TurboQuant** achieves near-optimal distortion at b bits per coordinate
**without training codebooks**, by first rotating the space with a
randomized Hadamard transform to make coordinates approximately
Gaussian, then quantizing each coordinate independently with the
optimal scalar quantizer for N(0,1).

> 📄 **Paper:** Zandieh, Daliri, Hadian, Mirrokni (2025).
> *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.*
> ICLR 2026 — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
>
> `snapvec` is an independent, pure-NumPy implementation of the algorithm
> described in that paper. If you use snapvec in academic work, please
> cite the TurboQuant paper.

---

### Algorithm

#### Step 1 — Normalize

Given a raw embedding `v ∈ ℝᵈ`, compute the unit vector `v̂ = v / ‖v‖` and store
`‖v‖` separately (float32, 4 bytes per vector).

#### Step 2 — Randomized Hadamard Transform (RHT)

Pad `v̂` to the next power of 2 (`d' = 2^⌈log₂ d⌉`), then apply:

```
x = (1/√d') · H · D · v̂
```

where:
- `D = diag(σ₁, …, σ_d')` — diagonal matrix of i.i.d. ±1 random signs (seed-deterministic)
- `H` — unnormalized Walsh-Hadamard matrix (butterfly pattern)

By the Johnson-Lindenstrauss lemma, each coordinate `xᵢ ≈ N(0, 1/d')`.
After rescaling `x̃ = x · √d'`, the coordinates are approximately `N(0,1)`
regardless of the original distribution of `v`.

**Complexity:** O(d log d) — no matrix multiplication, no codebook training.

**Implementation note (v0.3.0):** the butterfly is fully vectorised via a
reshape view — each level becomes one pair of NumPy ops on the whole
array instead of a Python `for` loop over `n / (2h)` slices. The diff is
tiny and the speedup is ~24× on a single query at `padded_dim = 512`:

```python
# Before: Python loop over n/(2h) slices per level → O(d) dispatches
while h < n:
    for i in range(0, n, h * 2):
        a = x[..., i : i + h].copy()
        b = x[..., i + h : i + 2 * h]
        x[..., i : i + h]       = a + b
        x[..., i + h : i + 2 * h] = a - b
    h *= 2

# After: reshape view → one pair of ops per level → O(log d) dispatches
while h < n:
    view = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
    a = view[..., 0, :].copy()
    view[..., 0, :] += view[..., 1, :]      # view[0] ← a + b
    view[..., 1, :]  = a - view[..., 1, :]  # view[1] ← a - b
    h *= 2
```

`reshape` returns a view only when the array is C-contiguous, so writes
to `view` propagate back to `x` — no extra allocation per level beyond
the single `a.copy()`. In the RHT pipeline we enforce this explicitly
(via `np.ascontiguousarray` at the call site and a defensive assert
inside `_fwht_inplace`); `.astype` alone does not guarantee C-order
when the input is Fortran-contiguous.

#### Step 3 — Lloyd-Max scalar quantization

The optimal scalar quantizer for N(0,1) at b bits partitions ℝ into 2^b intervals
and assigns each the conditional mean as reconstruction value.
These boundaries and centroids are precomputed and hardcoded in `snapvec._codebooks`
(no scipy required at runtime):

| bits | levels | distortion (MSE) | bytes/coord |
|------|--------|------------------|-------------|
| 2    | 4      | 0.1175           | 0.25        |
| 3    | 8      | 0.0311           | 0.375       |
| 4    | 16     | 0.0077           | 0.50        |

The quantized vector is stored as a `uint8` index matrix, tightly
bit-packed to `b/8` bytes per coordinate both on disk and in RAM. For 3-bit
this means packing 8 indices into 3 bytes (24 bits); for 2-bit and 4-bit
the packing is byte-aligned (4 per byte, 2 per byte respectively).

#### Step 4 — Approximate inner product

At search time the query `q` is rotated (not quantized) and the approximate
cosine similarity is computed as:

```
score(q, v) = (1/d') · Σᵢ centroid[idx_qᵢ] · centroid[idx_vᵢ]
```

This is a single float16 matrix–vector product against the cached centroid expansions.

---

### TurboQuant_prod: unbiased estimator with QJL correction

The MSE quantizer introduces a small systematic downward bias. The `use_prod=True` mode
corrects this using a **Quantized Johnson-Lindenstrauss (QJL)** residual:

**Build time (per stored vector):**

1. Quantize at `(b-1)` bits MSE, compute residual `r = x̃ - x̃_MSE`
2. Store `sign(S·r)` as a 1-bit vector (int8 ±1 in practice),
   where `S ∈ ℝ^(d'×d')` is a fixed random Gaussian matrix
3. Store `‖r‖ / √d'` (one float32 per vector)

**Query time (correction term):**

```
correctionᵢ = √(π/2) / d' · ‖rᵢ‖ · dot(S·q̂, sign(S·rᵢ))
final_scoreᵢ = mse_scoreᵢ + correctionᵢ
```

This follows from Lemma 4 of Zandieh et al. (2025):
`E[sign(S·r)] = √(2/π) · S·r / ‖S·r‖`, giving an unbiased estimate of `⟨r, q̂⟩`.

**When to use `use_prod=True`:**
- When you need accurate inner product magnitudes (KV-cache, attention approximation)
- **Not** recommended for pure ranking/NNS — the added QJL variance degrades recall@k
  relative to MSE-only at equal total bits

---

### Compression ratios

Measured on `d = 384` (BGE-small); the RHT pads to the next power of 2 so
`padded_dim = 512`. All per-vector byte counts below include the padded
dimensions — they are the numbers you actually pay in RAM and on disk.
The trailing `+ 4` is the float32 norm; in `normalized=True` mode it is
still persisted (as 1.0) to keep the on-disk layout stable across modes.

| Backend        | Bytes/vec (disk) | Bytes/vec (RAM, idle) | Disk ratio | RAM ratio |
|----------------|------------------|-----------------------|------------|-----------|
| float32        | 1 536            | 1 536                 | 1.0×       | 1.0×      |
| 4-bit snapvec  | 256 + 4          | 256 + 4               | **5.9×**   | **5.9×**  |
| 3-bit snapvec  | 192 + 4          | 192 + 4               | **7.8×**   | **7.8×**  |
| 2-bit snapvec  | 128 + 4          | 128 + 4               | **11.6×**  | **11.6×** |
| int8 (naïve)   | 384 + 4          | 384 + 4               | 4.0×       | 4.0×      |

**RAM = disk:** indices are tightly bit-packed in both. The same byte
layout is used everywhere, so `save` / `load` copy bytes directly without
an intermediate unpack/repack step. This halves indices RAM vs the
pre-v0.2 behaviour (which stored uint8 in RAM and only packed on disk).

Bit-packing scheme:
- **4-bit** (`0.5 bytes/coord`): 2 indices per byte (byte-aligned).
- **3-bit** (`0.375 bytes/coord`): 8 indices → 3 bytes, cross-byte tight
  packing (v3 file format; v1/v2 files are read transparently via the
  legacy byte-aligned decoder).
- **2-bit** (`0.25 bytes/coord`): 4 indices per byte (byte-aligned).

Unpacking happens when the `float16` centroid cache is built (cached
full-scan path — once, off the hot matmul) or per-chunk/per-query in
`chunk_size` / `filter_ids` modes.

**Search cache:** a lazy `float16` centroid expansion of shape
`(N, padded_dim)` is materialised on first query for fast matmul (~5 ms
at N = 100 k). It is evicted on writes and can be avoided entirely via
`chunk_size` for memory-constrained deployments.

#### Real-world footprint: 1 M vectors at d = 768

Typical for BGE-base, E5-base, `nomic-embed-text-v1`, and other 768-dim
models. The RHT pads to 1024.

| Backend        | Idle RAM | + cache (float16) | Warm peak |
|----------------|----------|--------------------|-----------|
| float32        | **2.86 GiB** | — | 2.86 GiB |
| int8 (naïve)   | 0.72 GiB | — | 0.72 GiB |
| 4-bit snapvec  | **0.48 GiB** | +1.91 GiB | 2.39 GiB |
| 3-bit snapvec  | **0.36 GiB** | +1.91 GiB | 2.27 GiB |
| 2-bit snapvec  | **0.24 GiB** | +1.91 GiB | 2.15 GiB |

Numbers use binary units (1 GiB = 2³⁰ bytes); e.g. float32 is
`1M × 768 × 4 B = 2.86 GiB`.

The cache is materialised only during active search and is evicted on
any write. With `chunk_size` set, warm peak drops to roughly
`idle RAM + chunk_size × padded_dim × 2 B` (the per-chunk float16
scratch) — e.g. `chunk_size=10_000` at `padded_dim=1024` adds ~20 MiB,
at the cost of ~10× query latency. This is the usual memory/latency
trade-off, exposed as a first-class flag.

For a single-server RAG index at this scale, 4-bit snapvec idles at
**half a GiB** where float32 idles at ~3 GiB.

---

### Recall benchmarks

Measured on synthetic unit-sphere vectors (`d=384`, `N=10 000`, 100 queries).
**Baseline: exact cosine float32 brute-force.**

| bits | recall@1 | recall@10 | recall@50 |
|------|----------|-----------|-----------|
| 2    | 0.72     | 0.83      | 0.91      |
| 3    | 0.81     | 0.91      | 0.96      |
| 4    | 0.86     | 0.93      | 0.95      |

Recall improves with clustered (real-world) data. On BGE-small-en embeddings
from mixed document corpora, 4-bit achieves **recall@10 ≈ 0.95**.

> **Note on published results:** The TurboQuant paper (Zandieh et al., 2025) reports
> recall up to 0.99, measured against HNSW graph navigation (not brute-force float32),
> on GloVe `d=200` data, using recall@1 with large `k_probe`. These conditions differ
> from the above; both results are correct under their respective definitions.

---

### File format (`.snpv`)

```
Offset  Size   Field
──────────────────────────────────────────────────
0       4 B    magic: "SNPV"
4       4 B    version: uint32 (1, 2, or 3)
8       4 B    dim: uint32  — original embedding dimension
12      4 B    bits: uint32 — total bits (2, 3, or 4)
16      4 B    seed: uint32 — rotation seed
20      4 B    n: uint32    — number of stored vectors
24      4 B    flags: uint32 — bit-0: use_prod, bit-1: normalized  [v2/v3 only]
──────────────────────────────────────────────────
28      4 B    packed_len: uint32
32      *      indices: bit-packed uint8 MSE indices
       n×4 B   norms: float32 per-vector original norms
[prod only]
       n×d' B  qjl_signs: int8 sign(S·r) per vector
       n×4 B   rnorms: float32 ‖r‖/√d per vector
──────────────────────────────────────────────────
       n×(2+L) ids: uint16-length-prefixed UTF-8 strings
```

Saves are **atomic** on POSIX: writes to `.snpv.tmp` then `os.replace()`.
Backward compatible: v1 (mse-only, pre-flags), v2 (flags + byte-aligned
3-bit), and v3 (tight 3-bit packing) files all load correctly — the
reader dispatches the 3-bit decoder on version.

---

## API reference

### `SnapIndex(dim, bits=4, seed=0, use_prod=False, chunk_size=None, normalized=False)`

| Parameter     | Type       | Default | Description |
|---------------|------------|---------|-------------|
| `dim`         | int        | —       | Embedding dimension |
| `bits`        | int        | 4       | Bits per coordinate: 2, 3, or 4 |
| `seed`        | int        | 0       | Rotation seed — must be consistent across build and query |
| `use_prod`    | bool       | False   | Enable QJL unbiased estimator (requires bits ≥ 3) |
| `chunk_size`  | int \| None | None    | Stream search in chunks without the float16 cache (for N > 500k) |
| `normalized`  | bool       | False   | Skip norm computation: trust that input vectors are unit-length |

### Methods

```python
idx.add(id, vector)                           # Add one vector
idx.add_batch(ids, vectors)                   # Add N vectors (~50x faster than loop)
idx.delete(id) -> bool                        # Remove by id, O(1) lookup
idx.search(query, k=10, filter_ids=None)      # [(id, score), ...] descending
idx.save(path)                                # Atomic binary save to .snpv
SnapIndex.load(path)                          # Load from .snpv file
idx.stats() -> dict                           # Compression / memory diagnostics
len(idx)                                      # Number of stored vectors
repr(idx)                                     # SnapIndex(dim=384, bits=4, mode=mse, n=1000)
```

> **All vector inputs should be `np.float32`.** Passing `float64` triggers
> a full-size temporary cast inside `add_batch` / `search` (see "Input
> dtype" in Quick start). Most embedding models already return `float32`;
> only ad-hoc arrays built from Python lists via `np.array(...)` default
> to `float64`.

---

## Relation to TurboQuant / PolarQuant

`snapvec` implements the core compression pipeline from:

> Zandieh, A., Daliri, M., Hadian, A., & Mirrokni, V. (2025).
> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.**
> *ICLR 2026.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

The same algorithm was published concurrently as "PolarQuant" at AISTATS 2026.
Both names were already taken on PyPI; `snapvec` = **Hada**mard + Lloyd-**Max**,
named after its two core operations.

Key contributions of this implementation over the reference:

- **No scipy** — codebooks hardcoded, numpy is the only runtime dependency
- **Batch WHT** — single O(n·d·log d) call for bulk inserts (~50x faster than loop)
- **Float16 cache** — centroid expansions in half precision, ~2x faster matmul
- **Packed RAM indices** — 2/4-bit indices stored bit-packed in RAM (2× less memory),
  unpacked lazily when the float16 cache is built — zero impact on warm query latency
- **Pre-filtering** — `filter_ids` restricts search to a subset with O(|filter| · d) cost
- **Chunked streaming search** — `chunk_size` avoids the float16 cache for large N
- **O(1) delete** — `_id_to_pos` dict + position compaction
- **Atomic saves** — `.snpv.tmp` → `os.replace()` pattern
- **Versioned format** — v1/v2/v3 all loadable, forward-compatible flags field,
  legacy 3-bit decoder kept around for pre-v0.3 indices

---

## Roadmap

The current implementation comfortably hits the 10–20 ms interactive budget
for N ≤ 200 k, d ∈ {384, 768}, with p50 warm-query latency of ~5 ms at
N=100 k, d=384. The float16 centroid cache carries ~77% of that time as a
BLAS-bound `gemv`; the remaining 23% is Python glue (RHT, normalization,
argpartition, result assembly).

Future work is scoped around this measured profile — we don't optimize
what's already fast.

### Pure-Python improvements (no new dependencies)

- **Quantized norms on disk** — store per-vector norms as uint8 with a
  (min, max) header. Saves 3 bytes/vec on disk with <0.1% precision loss.
- **Typed ID storage** — persist integer IDs as fixed-width uint32/uint64
  instead of UTF-8 strings when all IDs are numeric. Saves ~3 bytes/vec
  on disk for sequential integer IDs.
- **Positional-ID fast path** — when IDs are `0..N-1`, skip `_ids` list
  and `_id_to_pos` dict entirely; position is the ID. Saves ~23 bytes/vec
  of Python object overhead.

_Previously planned and now shipped:_
_Tight 3-bit packing (v0.3.0); vectorized FWHT (v0.3.0, ~25× faster per
query in isolation, ~10% E2E)._

### Hybrid Python + Rust core (opt-in accelerator)

The following phases would ship as an optional `snapvec-core` wheel. The
pure-Python path remains fully functional — Rust is detected at import
time and used automatically when available.

**Phase 1 — Cold-start / cache build (highest ROI)**
  - Move centroid expansion + `float16` conversion to Rust with SIMD.
  - Target: cold first-query 150 ms → ~20 ms on N=100 k, d=384.
  - Smallest API surface; trivial Python fallback.

**Phase 2 — LUT-based scan (eliminates the float16 cache)**
  - Implement PQ-style Asymmetric Distance Computation: per-query LUT
    (16 entries × pdim) + SIMD gather over packed indices.
  - Target: warm query 5 ms → ~1.5–2 ms; cache RAM: 102 MB → 0.
  - Enables N > 500 k with flat RAM growth.

**Phase 3 — Batch RHT + quantization**
  - Rust FWHT + `searchsorted` for `add_batch`.
  - Target: `add_batch(100k, d=384)` 2.5 s → ~200 ms.
  - Lowest priority — indexing is typically amortized offline.

### Non-goals

- **Trained codebooks (PQ / OPQ / RaBitQ-trained)**. Keeps the "no training
  required" guarantee; the data-agnostic Lloyd-Max tables make snapvec
  safe to use without a representative training sample.
- **Graph indices (HNSW / NSG)**. Different trade-off space; snapvec
  targets flat indices where compression and predictable latency matter
  more than sub-linear scan.
- **GPU acceleration**. The cache-matmul path is memory-bound on modern
  CPUs; GPU would help only at very large N where network/transfer cost
  already dominates.

---

## Changelog

See [`CHANGELOG.md`](./CHANGELOG.md) for the per-release history. Recent
highlights:

- **v0.3.0** — Tight 3-bit packing (5.9× → 7.8× compression for 3-bit),
  vectorised FWHT (~24× faster single-query RHT), file format v3 with
  transparent v1/v2 backward-compat.
- **v0.2.0** — RAM-packed indices for 2/4-bit (idle RAM cut in half),
  `normalized=True` flag, measured-accurate compression docs.

## Installation

```bash
pip install snapvec
```

**Requirements:** Python >= 3.10, NumPy >= 1.24.  No other runtime dependencies.

For development:

```bash
git clone https://github.com/stffns/snapvec
cd snapvec
pip install -e .
pytest tests/ -v
```

---

## License

MIT © 2025 Jayson Steffens.

The TurboQuant algorithm is described in [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
by Zandieh et al. (Google Research / ICLR 2026). This package is an independent implementation.

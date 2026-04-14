# snapvec

**Fast compressed approximate nearest-neighbor search.  Pure NumPy.  No heavy dependencies.**

`snapvec` implements the TurboQuant compression pipeline — randomized Hadamard transform followed by optimal Gaussian scalar quantization (Lloyd-Max) — as a self-contained Python library for embedding vector search.  It achieves **8–12× compression** with **>0.92 recall@10** against float32 brute-force, using only NumPy.

```
pip install snapvec
```

---

## Quick start

```python
import numpy as np
from snapvec import SnapIndex

# Build index
idx = SnapIndex(dim=384, bits=4)          # 4-bit, ~8x compression
idx.add_batch(ids=list(range(N)), vectors=embeddings)

# Query
results = idx.search(query_vector, k=10)     # [(id, score), ...]

# Persist
idx.save("my_index.snpv")
idx2 = SnapIndex.load("my_index.snpv")   # atomic save, v1/v2 compatible
```

---

## Technical background

### The problem: embedding vectors are expensive

Modern embedding models produce float32 vectors of dimension `d ∈ {384, 768, 1536}`.
Storing N vectors requires `4·N·d` bytes; brute-force search costs `O(N·d)` per query.
For N = 1M, d = 384: **1.5 GB RAM**, with inner products dominating inference time.

Product Quantization (PQ) splits vectors into M sub-vectors and quantizes each
independently. It is effective but requires training a K-means codebook per dataset.
Random Binary Quantization (RaBitQ, 1-bit) is fast but coarse.

**TurboQuant** (Zandieh et al., ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
achieves near-optimal distortion at b bits per coordinate **without training codebooks**,
by first rotating the space with a randomized Hadamard transform to make coordinates
approximately Gaussian, then quantizing each coordinate independently with the
optimal scalar quantizer for N(0,1).

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

#### Step 3 — Lloyd-Max scalar quantization

The optimal scalar quantizer for N(0,1) at b bits partitions ℝ into 2^b intervals
and assigns each the conditional mean as reconstruction value.
These boundaries and centroids are precomputed and hardcoded in `snapvec._codebooks`
(no scipy required at runtime):

| bits | levels | distortion (MSE) | bytes/coord (disk) |
|------|--------|------------------|--------------------|
| 2    | 4      | 0.1175           | 0.25               |
| 3    | 8      | 0.0311           | 0.375              |
| 4    | 16     | 0.0077           | 0.50               |

The quantized vector is stored as a `uint8` index matrix, bit-packed to `b/8`
bytes per coordinate on disk.

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

For N vectors of dimension `d = 384` (BGE-small):

| Backend        | Bytes/vector (disk) | Bytes/vector (RAM, idle) | Ratio vs float32 |
|----------------|---------------------|--------------------------|------------------|
| float32        | 1 536               | 1 536                    | 1.0×             |
| 4-bit snapvec  | 192 + 4             | 256 + 4                  | **7.9× / 5.9×**  |
| 3-bit snapvec  | 144 + 4             | 512 + 4                  | **10.4× / 3.0×** |
| 2-bit snapvec  | 96 + 4              | 128 + 4                  | **15.4× / 11.6×**|
| int8 (naïve)   | 384 + 4             | 384 + 4                  | 3.9×             |

The 4-byte overhead is the per-vector norm. **RAM indices are bit-packed** when
`bits` evenly divides 8 (2-bit: 4 per byte, 4-bit: 2 per byte); 3-bit falls back
to uint8 storage. Bit-packing also applies on disk for all bit widths.

During active search, a lazy `float16` centroid cache (`2·padded_dim` bytes/vec)
is materialised for fast matmul; it is evicted on writes and can be avoided
entirely via `chunk_size` for memory-constrained deployments.

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
0       4 B    magic: "HDMX"
4       4 B    version: uint32 (1 or 2)
8       4 B    dim: uint32  — original embedding dimension
12      4 B    bits: uint32 — total bits (2, 3, or 4)
16      4 B    seed: uint32 — rotation seed
20      4 B    n: uint32    — number of stored vectors
24      4 B    flags: uint32 — bit-0: use_prod, bit-1: normalized  [v2 only]
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
Backward compatible: v1 files (mse-only) load correctly in any version.

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
- **Versioned format** — v1/v2 both loadable, forward-compatible flags field

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

- **Vectorized FWHT** — replace the Python-level butterfly loop in
  `_rotation._fwht_inplace` with reshape-based NumPy operations.
  Expected: ~10–20× speedup on the RHT step (0.47 ms → ~0.03 ms at d=512),
  ~5–10% end-to-end query improvement.
- **Quantized norms on disk** — store per-vector norms as uint8 with a
  (min, max) header. Saves 3 bytes/vec on disk with <0.1% precision loss.
- **Typed ID storage** — persist integer IDs as fixed-width uint32/uint64
  instead of UTF-8 strings when all IDs are numeric. Saves ~3 bytes/vec
  on disk for sequential integer IDs.
- **Positional-ID fast path** — when IDs are `0..N-1`, skip `_ids` list
  and `_id_to_pos` dict entirely; position is the ID. Saves ~23 bytes/vec
  of Python object overhead.

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

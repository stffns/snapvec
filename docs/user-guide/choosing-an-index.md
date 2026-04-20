# Choosing an index

| If you... | Pick | Why |
|-----------|------|-----|
| Just want a dependency-minimal ANN with no training | **`SnapIndex`** | Works on any distribution, no `fit` call. 4-bit default is a good balance. |
| Need recall above 0.95 without a training pass | **`ResidualSnapIndex`** | Two-stage scalar quantization + optional rerank, still training-free. |
| Have a corpus sample and want aggressive compression | **`PQSnapIndex`** | Learned codebooks reach 16-32 B/vec with recall far above scalar at matched bytes. |
| Need sub-linear search above ~100k vectors | **`IVFPQSnapIndex`** | Partitioned search visits `nprobe / nlist` of the corpus. With `rerank_candidates` it breaks the PQ recall ceiling. |

## Sizing rules of thumb

| Parameter | Required? | Guidance |
|-----------|-----------|----------|
| `bits` (SnapIndex) | no, defaults to 4 | 4 for ~95% recall at 6x compression. 3 for a sweet spot around 7.8x. 2 only for huge corpora where scale beats precision. |
| `M` (PQ) | **yes** | Higher M = higher recall, more disk. Starting point `dim // 4` (e.g. `M=96` for `dim=384`); many users ship with `M=16-32` for aggressive compression. |
| `K` (PQ) | no, defaults to 256 | Leave at 256 (one byte per sub-index). |
| `nlist` (IVF) | **yes** | Target `4 * sqrt(N)`. E.g. N=57k -> nlist=512, N=1M -> nlist=4096. |
| `nprobe` (IVF) | no, defaults to `nlist // 16` | Trades recall for latency; tune per query. |
| `rerank_candidates` (IVF) | no, defaults to `None` | Pass `100` to rerank the PQ candidates with the stored fp16 vectors. Raises recall toward the float32 ceiling. Requires `keep_full_precision=True` at construction. |

## Bits vs recall (SnapIndex)

| `bits` | Compression vs float32 | Recall@10 on real embeddings |
|--------|------------------------|-------------------------------|
| 2 | 11.6x | ~0.83 (synthetic), higher on clustered real data |
| 3 | 7.8x | ~0.92 |
| 4 | 5.9x | ~0.95 |

## When does IVF-PQ pay off?

- **N < 10k**: `SnapIndex` or `PQSnapIndex` full-scan is fine.
- **N = 50k-100k**: `PQSnapIndex` + scan or `IVFPQSnapIndex` -- both feasible.
- **N >= 100k**: `IVFPQSnapIndex` is the clear winner.
- **N >= 500k**: `IVFPQSnapIndex` is effectively required -- float32
  brute-force starts hitting RAM and latency walls.

See the [benchmarks](../benchmarks.md) page for measured numbers.

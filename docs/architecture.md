# Architecture

How `snapvec` compresses vectors without destroying inner-product recall.

## Motivation

Dense embeddings from modern encoders (384-dim BGE, 768-dim E5, 1024-dim+
OpenAI, etc.) are expensive to store and scan at scale. At float32,
1M vectors of dim=768 take 3 GB. Float16 halves that but still scales
linearly with dim and only halves latency because memory bandwidth is
the bottleneck.

`snapvec` targets the compression / recall frontier: give up a small
amount of recall (typically 1-5 percentage points) for 6-96x less disk
and RAM, and run search kernels that fit in L2 cache.

## Pipeline (SnapIndex)

1. **Normalize** (optional): unit-length every vector. Turns inner
   product into cosine similarity and removes one nuisance dimension.
2. **Randomized Hadamard Transform (RHT)**: multiply by a signed random
   Hadamard matrix. Decorrelates coordinates so their marginal
   distribution approaches iid Gaussian, even when the input is not.
3. **Lloyd-Max scalar quantization**: quantize each coordinate to `bits`
   levels using the Lloyd-Max boundaries for a unit Gaussian. This is
   the optimal scalar quantizer for a Gaussian source under MSE.
4. **Bit-pack** the quantized codes so storage is exactly `bits * pdim / 8`
   bytes per vector (`pdim` is the Hadamard-padded dim).

At query time, the query goes through the same normalize + RHT + quantize
path, then a vectorized inner-product sum over the unpacked codes scores
the whole corpus.

## Why RHT?

Scalar quantization is optimal for a Gaussian source. Real embeddings are
not Gaussian along each coordinate. RHT is an isometry that makes
coordinates look Gaussian without changing inner products, so Lloyd-Max
becomes near-optimal. It also distributes signal energy across
coordinates, so no single quantization error dominates.

The RHT is implemented as a reshape-view + in-place xor across `log2(pdim)`
levels, so it costs `O(pdim * log(pdim))` time but `O(log(pdim))` Python
dispatch overhead.

## Lloyd-Max codebooks

Pre-computed per bit-depth:

- 2-bit: 4 reconstruction levels at Gaussian-optimal boundaries
- 3-bit: 8 levels
- 4-bit: 16 levels

Stored as constants in [`snapvec/_codebooks.py`](https://github.com/stffns/snapvec/blob/main/snapvec/_codebooks.py).

## Product Quantization (PQSnapIndex)

For higher recall at the same bytes/vec, `PQSnapIndex` replaces scalar
quantization with product quantization:

1. Split each vector into `M` contiguous sub-vectors.
2. For each subspace, learn `K=256` k-means centroids over a training
   sample.
3. Encode each vector as `M` bytes (one centroid index per subspace).

Query scoring uses an ADC (asymmetric distance computation) lookup table:
pre-compute `M * K` partial inner products against the query, then sum
`M` table lookups per corpus vector. The Cython+OpenMP kernel
(`snapvec._fast.adc_colmajor`) runs this in parallel with scores as a
stripe of ~4-8 MB that fits in L2.

## Inverted File (IVFPQSnapIndex)

For sub-linear search, layer an inverted file (IVF) on top of PQ:

1. Learn `nlist` coarse centroids from a training sample.
2. Assign each corpus vector to its nearest coarse centroid.
3. Store vectors contiguous per cluster, with an `offsets` array for
   random access.
4. At query time, rank coarse centroids by query-centroid distance and
   visit only the top `nprobe` clusters.

Each cluster stores PQ codes of the **residual** (vector minus its coarse
centroid), which is easier to quantize than the raw vector.

### Float16 rerank

PQ has a recall ceiling set by codebook granularity. To break it,
`IVFPQSnapIndex` with `keep_full_precision=True` stores each vector in
float16 as well. At query time, the top `rerank_candidates` returned by
the PQ pass are rescored against their stored fp16 vectors, and the
final top-k comes from the rerank scores.

This recovers 4-7 percentage points of recall at the cost of one
`(rerank_candidates, dim) @ (dim,)` matmul per query, typically under
1 ms even at large `nprobe`.

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Jegou et al. (2011). *Product Quantization for Nearest Neighbor Search.*
  IEEE TPAMI.
- Andre et al. (2015). *Cache Locality is Not Enough: High-performance
  Nearest Neighbor Search with Product Quantization Fast Scan.*

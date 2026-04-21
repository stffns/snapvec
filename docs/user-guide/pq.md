# PQSnapIndex

Train-once product quantization. Learns per-subspace k-means codebooks
that adapt to the corpus distribution. Delivers 15-18 percentage points
higher recall@10 than `SnapIndex` at matched bytes/vec on modern LLM
embeddings, and opens ultra-compressed modes (16 / 32 / 64 B/vec) that
scalar quantization cannot reach.

## When to use

- You have access to a representative corpus sample (~10-50k rows).
- Recall target is above 0.95.
- You want aggressive compression (16-32 B/vec).

## Basic usage

```python
import numpy as np
from snapvec import PQSnapIndex

corpus = np.random.randn(50_000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

idx = PQSnapIndex(dim=384, M=16, K=256, seed=0)
idx.fit(corpus[:10_000])         # train codebooks on a sample
idx.add_batch(list(range(50_000)), corpus)

hits = idx.search(query, k=10)
```

## Choosing `M`

- `M` must divide the effective dim (or `pdim` if `use_rht=True`).
- Storage = `M` bytes/vec when `normalized=True` (plus 4 bytes otherwise
  for the stored norm).
- For `dim=384` (BGE-small): `M=16` gives 16 B/vec (24x compression vs
  float32); `M=32` gives 32 B/vec and ~2pp more recall.

## Why `use_rht=False` by default

`PQSnapIndex` learns codebooks directly on the embedding space, so the
RHT step (which decorrelates coordinates for scalar quantization) usually
hurts more than it helps: the k-means fit is already capturing
subspace-local structure. Pass `use_rht=True` only if your embeddings are
very non-gaussian and you see recall drop on held-out queries.

## OPQ rotation

`use_opq=True` turns on a learned orthogonal rotation during `fit()`
that balances per-subspace variance, typically lifting recall@10
by several percentage points at the same bytes/vec.  See the
[benchmarks](../benchmarks.md) page for the measured recall-vs-M
table on BEIR FIQA.

```python
idx = PQSnapIndex(dim=384, M=48, K=256, use_opq=True)
idx.fit(train_sample)
```

Cost: one extra eigendecomposition at fit time (milliseconds on
typical training sets), one extra `(dim, dim)` matmul per query
(microseconds).  Mutually exclusive with `use_rht`.

## File format

On-disk extension: `.snpq`. Magic `SNPQ`, v1 as of v0.9.0.

## API

See [`PQSnapIndex` API reference](../api/pq.md).

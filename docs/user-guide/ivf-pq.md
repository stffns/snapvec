# IVFPQSnapIndex

Sub-linear search at scale, with optional float16 rerank. Inverted-file
coarse partition on top of residual PQ: each query visits only
`nprobe / nlist` of the corpus. With `keep_full_precision=True` +
`rerank_candidates`, the top candidates are re-scored against the stored
float16 vectors, which recovers almost all the recall lost to PQ.

## Headline numbers

On BGE-small / FIQA (N = 57,638, dim = 384):

- recall@10 = **0.977 at 441 us / query**
- recall@10 = **0.998 at 1021 us / query**

5.8x faster than v0.6 at identical recall, past the PQ-only 0.929 ceiling.

## When to use

- N >= 100k -- below that, full-scan `PQSnapIndex` is comparable and simpler.
- Latency budget in the sub-millisecond range.
- Recall target >= 0.97.

## Basic usage

```python
import numpy as np
from snapvec import IVFPQSnapIndex

corpus = np.random.randn(100_000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

idx = IVFPQSnapIndex(
    dim=384,
    nlist=512,          # 4 * sqrt(N)
    M=16,
    K=256,
    keep_full_precision=True,
    seed=0,
)
idx.fit(corpus[:20_000])
idx.add_batch(list(range(100_000)), corpus)

hits = idx.search(query, k=10, nprobe=32, rerank_candidates=100)
```

## Sizing

| Parameter | Guidance |
|-----------|----------|
| `nlist` | `4 * sqrt(N)`; clamp between 32 and 65536 |
| `nprobe` | Start at `nlist // 16`; sweep to tune recall vs latency |
| Training set size | `>= 30 * nlist` rows (FAISS rule of thumb) |
| `rerank_candidates` | `None` for PQ-only; `100` for strong recall lift |
| `keep_full_precision` | `True` to enable `rerank_candidates` |

## Operating points (FIQA, BGE-small, N=57k)

| `nprobe` | `rerank_candidates` | recall@10 | latency |
|---------|---------------------|-----------|---------|
| 8 | None | 0.85 | 180 us |
| 32 | None | 0.92 | 340 us |
| 64 | 100 | 0.977 | 441 us |
| 256 | 200 | 0.998 | 1021 us |

See [benchmarks](../benchmarks.md) for the full sweep and reproduction
instructions.

## File format

On-disk extension: `.snpi`. Magic `SNPI`, v4 as of v0.9.0 (adds
float16 rerank cache).

## API

See [`IVFPQSnapIndex` API reference](../api/ivf-pq.md).

# SnapIndex

Training-free scalar-quantized index. Implements
[TurboQuant](https://arxiv.org/abs/2504.19874): randomized Hadamard
transform followed by Lloyd-Max scalar quantization. Works out of the
box on any vector distribution, no calibration or corpus sample required.

## When to use

- You don't want to run a one-off `fit` step.
- `dim * N` fits comfortably in RAM even at float32.
- Recall target is 0.92-0.95 at 6-12x compression.

## Basic usage

```python
import numpy as np
from snapvec import SnapIndex

corpus = np.random.randn(10_000, 384).astype(np.float32)
idx = SnapIndex(dim=384, bits=4, seed=0)
idx.add_batch(list(range(10_000)), corpus)

query = np.random.randn(384).astype(np.float32)
hits = idx.search(query, k=10)
```

!!! tip "`normalized` is an optimization, not a default"
    If your embeddings are already unit-length (for example, cosine-space
    outputs from most modern sentence encoders), pass `normalized=True`
    to skip the internal L2 normalization step.  With raw vectors (like
    the example above), leave it at the default `False`.  Passing
    `normalized=True` on non-unit inputs silently skips normalization
    and scores will not match cosine similarity.

## Bits guidance

Pick 4-bit unless you have a specific reason:

| `bits` | Compression | Recall@10 on real embeddings | Notes |
|--------|-------------|------------------------------|-------|
| 2 | 11.6x | ~0.83 | Only for aggressive compression |
| 3 | 7.8x | ~0.92 | Middle ground; tightly packed since v0.3 |
| 4 | 5.9x | ~0.95 | Default, recommended |

## Unbiased-estimator mode

For use cases that need unbiased inner-product estimates (KV-cache,
attention), pass `use_prod=True`. Applies the QJL correction at the cost
of roughly 2x search latency:

```python
idx = SnapIndex(dim=dim, bits=3, use_prod=True)
```

## File format

On-disk extension: `.snpv`. CRC32-checksummed, atomic writes (temp file
+ rename).

## API

See [`SnapIndex` API reference](../api/snap-index.md).

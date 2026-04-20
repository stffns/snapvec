# ResidualSnapIndex

Training-free, two-stage scalar quantization. Cascades a coarse Lloyd-Max
quantizer and a second-stage residual quantizer to reach operating points
that `SnapIndex` alone cannot (5-7 bits/coord with recall up to 0.96).

Exposes a **coarse-pass + rerank** search mode that converges to full-
reconstruction recall at `O(rerank_M)` candidates instead of `O(N)`;
`rerank_M = 100` already saturates on the tested corpora.

## When to use

- You need recall above 0.95 without running a corpus `fit`.
- You prefer scalar quantization to PQ for simplicity / portability.

## Basic usage

```python
import numpy as np
from snapvec import ResidualSnapIndex

corpus = np.random.randn(10_000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

idx = ResidualSnapIndex(dim=384, b1=3, b2=3, seed=0)
idx.add_batch(list(range(10_000)), corpus)

hits = idx.search(query, k=10, rerank_M=100)
```

## File format

On-disk extension: `.snpr`. CRC32-checksummed.

## API

See [`ResidualSnapIndex` API reference](../api/residual.md).

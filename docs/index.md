# snapvec

**Fast compressed approximate nearest-neighbor search. NumPy + Cython compiled kernels.**

`snapvec` ships four index types for embedding vector search, each targeting
a different point on the accuracy / storage / latency frontier:

| Index | Training | Compression | Recall | Use when |
|-------|----------|-------------|--------|----------|
| [`SnapIndex`](user-guide/snap-index.md) | none | 6-12x | 0.92+ | Any distribution, no corpus sample |
| [`ResidualSnapIndex`](user-guide/residual.md) | none | 4-8x | 0.96 | Higher recall, still training-free |
| [`PQSnapIndex`](user-guide/pq.md) | one-off `fit` | 24-96x | 0.95 | Modern LLM embeddings, aggressive compression |
| [`IVFPQSnapIndex`](user-guide/ivf-pq.md) | one-off `fit` | 24-96x | 0.98 | Sub-linear search at scale (N > 100k) |

All four file formats (`.snpv` / `.snpq` / `.snpr` / `.snpi`) carry a CRC32
trailer -- silent disk or transport corruption is caught at `load()` time
instead of returning wrong results.

## Install

```bash
pip install snapvec
```

## Quickstart

```python
import numpy as np
from snapvec import SnapIndex

idx = SnapIndex(dim=384, bits=4, normalized=True)
idx.add_batch(list(range(10_000)), np.random.randn(10_000, 384).astype(np.float32))

results = idx.search(np.random.randn(384).astype(np.float32), k=10)
for doc_id, score in results:
    print(doc_id, score)
```

See [Quickstart](getting-started/quickstart.md) for the end-to-end tour and
[Choosing an index](user-guide/choosing-an-index.md) for the decision tree.

## Context

`snapvec` was developed as the quantization layer for
[vstash](https://github.com/stffns/vstash), a local-first hybrid retrieval
system, to extend it to corpora beyond the float32 memory budget while
preserving its dependency-minimal design. It stands alone as a quantization
library, but the design constraints (NumPy-only base install, predictable
latency, reproducible index files) come from vstash's local-first
requirements.

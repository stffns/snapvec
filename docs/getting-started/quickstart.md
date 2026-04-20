# Quickstart

Five-minute tour. Run [`examples/quickstart.py`](https://github.com/stffns/snapvec/blob/main/examples/quickstart.py)
end-to-end for a working script.

## 1. Build an index

`SnapIndex` is the simplest index: training-free scalar quantization on
top of the randomized Hadamard transform. No `fit` call needed.

```python
import numpy as np
from snapvec import SnapIndex

rng = np.random.default_rng(0)
corpus = rng.standard_normal((10_000, 384)).astype(np.float32)

idx = SnapIndex(dim=384, bits=4, normalized=True, seed=0)
idx.add_batch(list(range(10_000)), corpus)
```

`ids` can be any hashable (int, str, UUID); they are round-tripped through
save/load.

## 2. Query

```python
query = rng.standard_normal(384).astype(np.float32)
hits = idx.search(query, k=10)

for doc_id, score in hits:
    print(doc_id, score)
```

`search` returns `list[tuple[id, float]]` sorted by descending score.

## 3. Persist

```python
idx.save("my.snpv")
loaded = SnapIndex.load("my.snpv")
```

Writes are atomic (temp file + rename) and CRC32-checksummed.

## Next steps

- [Choosing an index](../user-guide/choosing-an-index.md) -- decision tree
  across the four index types.
- [User guide](../user-guide/snap-index.md) -- deep dives per index.
- [Architecture](../architecture.md) -- how the compression works.

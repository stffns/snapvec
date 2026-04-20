# snapvec

[![PyPI version](https://img.shields.io/pypi/v/snapvec.svg)](https://pypi.org/project/snapvec/)
[![Python versions](https://img.shields.io/pypi/pyversions/snapvec.svg)](https://pypi.org/project/snapvec/)
[![CI](https://github.com/stffns/snapvec/actions/workflows/ci.yml/badge.svg)](https://github.com/stffns/snapvec/actions/workflows/ci.yml)
[![Docs](https://github.com/stffns/snapvec/actions/workflows/docs.yml/badge.svg)](https://stffns.github.io/snapvec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/snapvec/month)](https://pepy.tech/project/snapvec)

**Fast compressed approximate nearest-neighbor search. NumPy + Cython compiled kernels.**

Four index types for embedding vector search, each targeting a different
point on the accuracy / storage / latency frontier:

| Index | Training | Compression | Recall | Use when |
|-------|----------|-------------|--------|----------|
| `SnapIndex` | none | 6-12x | 0.92+ | Any distribution, no corpus sample |
| `ResidualSnapIndex` | none | 4-8x | 0.96 | Higher recall, still training-free |
| `PQSnapIndex` | one-off `fit` | 24-96x | 0.95 | Modern LLM embeddings, aggressive compression |
| `IVFPQSnapIndex` | one-off `fit` | 24-96x | 0.98 | Sub-linear search at scale (N > 100k) |

Headline number: **recall@10 = 0.977 at 441 us/query** on BEIR FIQA
(N = 57,638, BGE-small), 25-125x faster than sqlite-vec at comparable
recall.

## Install

```bash
pip install snapvec
```

On macOS you also need `brew install libomp` to build from source; the
wheels on PyPI bundle it.

## Quickstart

```python
import numpy as np
from snapvec import SnapIndex

rng = np.random.default_rng(0)
corpus = rng.standard_normal((10_000, 384)).astype(np.float32)

idx = SnapIndex(dim=384, bits=4, seed=0)
idx.add_batch(list(range(10_000)), corpus)

query = rng.standard_normal(384).astype(np.float32)
for doc_id, score in idx.search(query, k=10):
    print(doc_id, score)

idx.save("my.snpv")
```

Runnable end-to-end scripts for every index live in
[`examples/`](examples/).

## Documentation

Full docs: **<https://stffns.github.io/snapvec/>**

- [Installation](https://stffns.github.io/snapvec/getting-started/installation/)
- [Choosing an index](https://stffns.github.io/snapvec/user-guide/choosing-an-index/)
- [Architecture](https://stffns.github.io/snapvec/architecture/) (RHT, Lloyd-Max, PQ, IVF)
- [Benchmarks](https://stffns.github.io/snapvec/benchmarks/)
- [API reference](https://stffns.github.io/snapvec/api/snap-index/)

## Context

`snapvec` was developed as the quantization layer for
[vstash](https://github.com/stffns/vstash), a local-first hybrid retrieval
system, to extend it to corpora beyond the float32 memory budget while
preserving its dependency-minimal design. It stands alone as a
quantization library, but the design constraints (NumPy-only base
install, predictable latency, reproducible index files) come from
vstash's local-first requirements.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test matrix,
and the release process. Bugs and feature requests go to
[issues](https://github.com/stffns/snapvec/issues); questions and
usage help to [discussions](https://github.com/stffns/snapvec/discussions).

## License

MIT (c) 2025 Jayson Steffens.

The TurboQuant algorithm is described in
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874) by Zandieh et al.
(Google Research / ICLR 2026). This package is an independent
implementation.

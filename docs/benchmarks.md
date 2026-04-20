# Benchmarks

All numbers measured on the same hardware with a fixed random seed. Raw
scripts live in [`experiments/`](https://github.com/stffns/snapvec/tree/main/experiments);
a reproducible CI-runnable suite is planned.

## Headline: IVFPQSnapIndex on FIQA

BEIR FIQA, N = 57,638, dim = 384 (BGE-small), nlist = 512, M = 192, K = 256.
`keep_full_precision=True`, `rerank_candidates=100`.

| `nprobe` | Configuration | recall@10 | Latency (us/query) |
|---------|---------------|-----------|--------------------|
| 8 | PQ only | 0.85 | 180 |
| 32 | PQ only | 0.92 | 340 |
| 64 | PQ + fp16 rerank | **0.977** | **441** |
| 256 | PQ + fp16 rerank | **0.998** | **1021** |

5.8x speedup vs v0.6 at identical recall, past the PQ-only 0.929 ceiling.

## snapvec vs sqlite-vec (measured)

Same corpus, same hardware. `snapvec` at `nprobe=64` + rerank.

| N | sqlite-vec | snapvec | Speedup | Recall tradeoff |
|---|------------|---------|---------|-----------------|
| 10k | 2.3 ms | 0.44 ms | 5x | 0.997 |
| 57k | 15.1 ms | 0.44 ms | 34x | 0.977 |
| 100k | 23.8 ms | 1.04 ms | 23x | 0.994 |
| 500k | ~110 ms | 0.9 ms | 125x | ~0.97 |
| 1M | brute-force infeasible | 1.1 ms | -- | -- |

Disk footprint is 2-8x smaller across the range.

## Compression ratios

For BGE-small (dim=384, float32 baseline = 1536 B/vec):

| Index | Config | B/vec | Compression |
|-------|--------|-------|-------------|
| `SnapIndex` | bits=2 | 132 | 11.6x |
| `SnapIndex` | bits=3 | 196 | 7.8x |
| `SnapIndex` | bits=4 | 260 | 5.9x |
| `PQSnapIndex` | M=16 | 16 | 96x |
| `PQSnapIndex` | M=32 | 32 | 48x |
| `IVFPQSnapIndex` | M=192 + fp16 rerank | ~960 | 1.6x (rerank cache dominates) |
| `IVFPQSnapIndex` | M=192, no rerank | ~192 | 8x |

## Reproduction

```bash
pip install -e ".[dev]"
python experiments/bench_v090_fiqa.py       # FIQA recall / latency
python experiments/bench_sqlite_vec_baseline.py   # sqlite-vec comparison
```

The `experiments/` folder is WIP; expect rough edges. A first-class
`bench/` suite that runs in CI and emits machine-readable results is
tracked on the roadmap.

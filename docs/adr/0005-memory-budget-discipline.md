# ADR 0005: Memory-budget discipline in fit and load

Date: 2026-04-22
Status: accepted

## Context

`snapvec` targets a laptop-local workflow: developer machines with
8-16 GB RAM, corpora in the 10k-1M vector range.  At the top of
that range, the difference between "peak memory equals final size"
and "peak memory equals 2x-10x the final size" is the difference
between a job that completes and a job that is killed by the OOM
killer or grinds through swap.

Two places in the library were violating this implicitly:

**OPQ fit.**  `fit_opq_rotation` used to cast the entire centred
float32 batch to float64 for covariance accumulation:

```python
X_c64 = X_c.astype(np.float64)   # n * d * 8 bytes transient
cov = (X_c64.T @ X_c64) / n
```

At N=1M, d=384 that is ~3 GB of transient memory on top of the
original float32 input.  On a 16 GB laptop with embeddings already
resident, `use_opq=True` could OOM silently.

**Load paths.**  All four index loaders shared the pattern:

```python
arr = np.frombuffer(f.read(n_bytes), dtype=...).reshape(...).copy()
```

`f.read` returns a `bytes` object that stays alive until the
assignment completes; `.copy()` allocates a second buffer of the
same size before `arr` takes ownership.  Peak memory during load
is ~2x the final array.  `PQSnapIndex.load()` additionally did a
`.T.copy()` to transpose the on-disk (n, M) row-major block into
the (M, n) C-contiguous RAM layout, adding another full-size
transient.

Individually each was a tolerable hiccup.  Cumulatively they meant
the library documented a 1M-vector corpus as supported while
quietly needing 4-5 GB of headroom above the final index size to
actually get there.

## Decision

Adopt a blanket rule for the fit and load paths: never hold two
copies of any per-corpus-sized array at the same time, and cap
transient memory at a constant multiple of the per-vector cost.

**`fit_opq_rotation` accumulates covariance in 16,384-row chunks.**

```python
cov = np.zeros((d, d), dtype=np.float64)
for start in range(0, n, chunk):
    X_chunk = X[start:start + chunk] - mean       # float32
    X_chunk64 = X_chunk.astype(np.float64)        # ~chunk * d * 8 bytes
    cov += X_chunk64.T @ X_chunk64
```

Peak transient is `chunk * d * 8` bytes (~30 MB at dim=384),
independent of `N`.  Numerics are unchanged: same per-row outer
products, same float64 accumulator, same eigendecomposition.

**`load()` uses `readinto` into pre-allocated numpy buffers.**

```python
arr = np.empty(shape, dtype=...)
f.readinto(arr.data)
```

No `bytes` transient, no `.copy()` duplicate.  `PQSnapIndex.load()`
additionally streams the on-disk (n, M) block through a 16,384-row
staging buffer and fuses the transpose into each chunk's copy into
the final (M, n) array, so the full-size transpose transient is
also gone.

## Consequences

- Measured on a clean subprocess (so `ru_maxrss` is not
  contaminated by earlier phases):

  | index | n    | M    | file   | codes  | peak RSS before | peak RSS after |
  |:------|-----:|-----:|-------:|-------:|----------------:|---------------:|
  | PQ    | 500k |  96  | 50 MB  | 46 MB  |      199 MB     |     152 MB     |
  | IVFPQ | 500k | 192  | 96 MB  | 92 MB  |      291 MB     |     198 MB     |

  Savings are linear in `N` -- one codes-array-worth of transient
  removed per load.

- OPQ fit at N=1M drops from ~3 GB transient peak to ~30 MB
  transient peak, independent of `N`.

- File formats are unchanged.  Existing `.snpv`, `.snpq`, `.snpi`,
  `.snpr` files round-trip identically under the new code.

- The rule generalises.  Any future feature that touches a
  per-corpus-sized array in the fit or load path should:

  1. Pre-allocate the final buffer and fill it (no
     `frombuffer + copy`, no `array + array.copy()`).
  2. Stream arithmetic that needs a different dtype through a
     small bounded chunk, not a whole-batch cast.
  3. Flag in review if these are not possible; do not ship a
     silent 2x-10x memory blow-up.

- The one remaining per-corpus allocation at load time is the
  final codes array itself.  Reducing that further requires a
  memory-mapped lazy path (v0.12 roadmap).  That is out of scope
  for this ADR.

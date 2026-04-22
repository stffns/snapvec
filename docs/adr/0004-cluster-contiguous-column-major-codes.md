# ADR 0004: Cluster-contiguous column-major codes in IVFPQ

Date: 2026-03 (v0.6)
Status: accepted

## Context

`IVFPQSnapIndex` stores per-vector PQ codes and needs to serve two
access patterns:

1. **Search.**  Visit the top `nprobe` clusters.  For each cluster,
   score every vector in it with the ADC kernel (`M` LUT lookups per
   candidate, summed into a score).
2. **Ingest.**  `add_batch` merges new vectors into the corpus,
   each assigned to its coarse cluster.

Two natural layouts, each broken for the other workload:

- **`(N, M)` row-major, unsorted.**  Each row is one vector's M
  codes.  Search has to build a boolean mask (`clusters == c`) or a
  fancy-index list per probed cluster, per query.  At N=1M and
  nprobe=16, this is 16 mask builds of size 1M per query, dominating
  the kernel time.
- **`(M, N)` column-major, unsorted.**  Each column is one subspace
  across all vectors.  The ADC kernel wants this for cache locality
  (one subspace scan per LUT entry), but search still has to mask
  per cluster.

Both layouts were measured in `experiments/bench_ivf_pq_contiguous.py`.
The mask-per-cluster path was the dominant cost at N >= 20k.

## Decision

Store codes as `(M, N)` uint8, **sorted by cluster id** at add
time, with an `offsets` array of length `nlist + 1` giving the
start and end row of each cluster's vectors.

Probe cluster `c` becomes:

```python
start, end = offsets[c], offsets[c + 1]
codes[:, start:end]        # contiguous column-block, one stride-1 slice
```

No mask, no fancy-index.  The ADC kernel walks a contiguous block
of bytes per subspace per cluster.

## Consequences

- Search is fast: each probed cluster is a contiguous slice.  The
  ADC kernel sees stride-1 reads across both M and the
  within-cluster N dimension.  This is what makes IVF pay off in
  wall-clock terms on top of the full-scan PQ baseline.
- Ingest is more expensive.  `add_batch` has to merge the new
  batch into the cluster-contiguous layout.  The current
  implementation sorts just the new batch by cluster id, then
  performs one per-cluster contiguous memcpy into a freshly
  allocated `(M, N + batch)` array.  That is O(N) memory and O(N)
  time per call.
- True O(1) per-row streaming requires a separate delta buffer
  layout.  This is the v0.12 roadmap item; until it lands, snapvec
  is positioned for corpora that are relatively static after an
  initial bulk load, not for live per-row ingest.
- The on-disk format stores codes in (M, N) column-major row-major
  order, matching the RAM layout, so `load()` goes straight into
  the right buffer with no transpose (see ADR 0005 for the PQ case
  where the transpose was unavoidable).
- The `(M, N)` layout is visible at the Python level via
  `IVFPQSnapIndex._codes.shape`.  External code that depends on
  this internal attribute is on its own -- it is not part of the
  public API.

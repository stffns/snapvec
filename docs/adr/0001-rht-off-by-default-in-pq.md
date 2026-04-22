# ADR 0001: RHT off by default in PQ indexes

Date: 2026-04-14
Status: accepted

## Context

`SnapIndex` (scalar quantisation) relies on a randomised Hadamard
transform (RHT) before Lloyd-Max quantisation because its codebooks
are trained against a standard Gaussian, and the RHT Gaussianises
arbitrary input distributions so the single 1D codebook is close to
optimal for every coordinate.  This is the core TurboQuant argument.

When we added `PQSnapIndex`, the natural reflex was to keep RHT on
for the same reason: more uniform per-coordinate variance means
codebook entries get used more evenly.

But PQ is not Lloyd-Max.  PQ partitions the (rotated) vector into
`M` subspaces of size `d_sub` and trains an independent k-means
codebook per subspace.  What PQ actually exploits is *structure*
within each subspace -- correlated coordinates that cluster tightly.
RHT mixes coordinates across the whole vector, so the post-RHT
subspaces are near-isotropic and the k-means codebooks have no
structure to compress.

Measured on BGE-small / SciFact and scaled up to FIQA
(`experiments/bench_pq_scaleup_validation.py`):

- PQ without RHT beats `SnapIndex(bits=3)` and `SnapIndex(bits=4)` at
  matched or lower storage across `K in {16, 64, 256}` and three
  seeds.
- PQ with RHT enabled loses ~10-15 percentage points of recall at
  the same bytes/vec on modern sentence embeddings.
- The gap is robust to K: larger K helps both variants but does not
  close the RHT/no-RHT delta.

## Decision

`PQSnapIndex(use_rht=False)` is the default.  The flag still exists
for compatibility and for distributions where RHT happens to help
(for example near-uniform synthetic noise), but users are not
steered toward it and the docstring flags the trade-off.

`IVFPQSnapIndex` inherits the same default because its residual
codebooks face the same argument.

OPQ (ADR 0003) is the recommended way to improve PQ recall in the
rotation family: it learns a *data-specific* rotation that
redistributes variance across subspaces without destroying the
per-subspace correlation structure the codebooks need.

## Consequences

- New users get the configuration that works on real embeddings out
  of the box.  The `use_rht=True` path is reachable but signposted.
- `use_opq` and `use_rht` are declared mutually exclusive at
  `__init__` -- mixing a random rotation with a learned one would
  waste the OPQ fit.
- The `use_rht=True` code path still has test coverage (round-trip,
  determinism) so it is safe to enable, just not recommended.
- A future experiment on a genuinely non-Gaussian synthetic corpus
  could revisit whether there is a regime where RHT-before-PQ pays
  off; the flag stays in the API so that experiment does not
  require a refactor.

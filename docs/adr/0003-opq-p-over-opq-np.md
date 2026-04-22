# ADR 0003: OPQ-P over OPQ-NP

Date: 2026-04-21
Status: accepted

## Context

OPQ (Ge et al., 2013) learns an orthogonal rotation that balances
per-subspace variance before PQ encoding.  Orthogonality preserves
inner products, so `<q, x> == <Rq, Rx>`, and the rest of the search
pipeline is unchanged.

The original paper proposes two variants:

- **OPQ-NP (non-parametric).**  Alternating optimisation: freeze R,
  refit the PQ codebooks on `X R`; freeze the codebooks, refit R as
  the orthogonal Procrustes solution.  Typically 5-10 outer
  iterations until convergence.  Best reported recall.
- **OPQ-P (parametric).**  Assume a centred Gaussian source.  The
  optimal R is the sorted eigenvector matrix of the covariance
  with a round-robin subspace allocation that spreads high-variance
  directions across subspaces.  One eigendecomposition, no outer
  loop.  Faster to fit, slightly lower recall.

On BEIR FIQA (BGE-small, dim=384) measured by us at three M values:

|  M  | d_sub | PQ baseline | OPQ-P | delta   |
|:---:|:-----:|:-----------:|:-----:|:-------:|
|  48 |   8   |    0.553    | 0.656 | +10.3pp |
|  96 |   4   |    0.767    | 0.812 |  +4.6pp |
| 192 |   2   |    0.932    | 0.931 |    0    |

Published OPQ-NP numbers on comparable setups report an additional
**+0.3-0.8 percentage points** on top of OPQ-P, at the cost of the
iterative fit loop: minutes instead of seconds at N = 100k-1M.

## Decision

Ship OPQ-P.  The flag is `use_opq=True` on both `PQSnapIndex` and
`IVFPQSnapIndex`, defaults to `False`.

`fit_opq_rotation` does:

1. Centre X (per-chunk, float32 arithmetic; ADR 0005).
2. Accumulate `(d, d)` covariance in float64 chunks.
3. `np.linalg.eigh` on the `(d, d)` covariance.
4. Sort eigenvectors by eigenvalue descending.
5. Round-robin to `M` subspaces of size `d / M`: eigenvector `i`
   goes to subspace `i % M`.  Spreads high-variance directions
   across subspaces.
6. Return the stacked column matrix as a `(d, d)` float32 rotation.

## Consequences

- `fit()` stays in the seconds-range for the sizes we target
  (10k-1M training rows).  A library positioned as "laptop-local
  ANN with predictable latency" would be poorly served by an
  iterative outer loop that turns fit into a multi-minute job.
- We leave 0.3-0.8 pp of recall on the table at configurations
  where OPQ already helps.  On configurations where OPQ-P gives
  zero gain (`d_sub < 4`), OPQ-NP also gives zero gain; neither
  variant has structure to exploit.
- The parametric assumption (centred Gaussian source) is violated
  on real embeddings, but in the regime where OPQ helps at all
  (`d_sub >= 4`), OPQ-P captures most of the available gain.
- OPQ-NP remains a possible future addition.  It would be a
  separate fit function, not a flag on `fit_opq_rotation`, because
  its outer loop interleaves with PQ codebook training.  If added,
  it supersedes neither this ADR nor the existing `use_opq`
  semantics -- it is a strictly additional option.
- Anyone tempted to "just add a few iterations" to the current
  parametric path should know that is not what makes OPQ-NP work.
  The non-parametric variant also refits the PQ codebooks at each
  iteration; a rotation-only loop converges to the OPQ-P solution.

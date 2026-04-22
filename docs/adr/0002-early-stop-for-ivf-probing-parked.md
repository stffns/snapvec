# ADR 0002: `early_stop` for IVF probing parked

Date: 2026-04-21
Status: accepted (negative result)

## Context

`IVFPQSnapIndex.search` visits the top `nprobe` clusters by coarse
score and runs the ADC kernel over each.  A classic speedup is to
short-circuit the probe loop once the next cluster's *best possible*
residual score (`coarse_dot + sum_j max_k lut[j, k]`) falls below
the current k-th best actual score: any remaining cluster is
provably dominated and can be skipped.

This is an upper-bound pruning argument, so it is strict: recall
does not drop.  If the bound is tight enough, latency goes down.

We implemented this in the `feat/ivfpq-early-stop` branch, batched
in chunks of 32 clusters to amortise the per-check dispatch cost,
and measured it on BEIR FIQA (N=57,638, BGE-small, M=192, K=256)
against the baseline batched full-scan.

| nprobe | full ms | early ms | speedup |
|-------:|--------:|---------:|--------:|
| 4      |  0.16   |   0.16   |  0.98x  |
| 32     |  0.34   |   0.55   |  0.62x  |
| 256    |  1.09   |   3.43   |  0.32x  |

Recall is identical at every `nprobe` (the bound is strict), but
`early_stop` is **slower** across the board.  Three independent
reasons, each by itself enough to kill the speedup:

1. The per-chunk `fused_gather_adc` dispatch and the Python-level
   merge cost already eat most of the baseline kernel's total
   time.  Skipping clusters late in the probe list does not claw
   back enough work to pay for the bookkeeping.
2. The global bound `sum_j max_k lut[j, k]` is loose on real
   embeddings.  Actual residual scores rarely approach the
   max-per-subspace product; most clusters that *could* be pruned
   by a tighter bound are not pruned by this one.
3. FIQA's recall curve is gradual (0.66 at `nprobe=4`, 0.93 at
   `nprobe=256`), which means top-k results are distributed across
   many clusters.  A workload with concentrated ground-truth in a
   small number of dominant clusters would benefit more.  FIQA is
   not that workload.

## Decision

Do not ship `early_stop`.  Keep the branch (`feat/ivfpq-early-stop`)
in git history for future reference but do not merge.

## Consequences

- Shipping-path code stays simpler: the search kernel has no mode
  switch for early termination.
- Recall guarantees are unchanged.
- A future contributor tempted to implement this again should first
  check whether at least one of the three failure modes above has
  been addressed:
  - A genuinely tighter per-cluster bound (for example, stored at
    `fit()` time as the max residual score for each cluster's
    training rows rather than a global product-of-maxes bound).
  - A workload with concentrated ground-truth clusters (benchmark
    on that, not on FIQA).
  - A kernel architecture where cluster-level dispatch is cheap
    enough that early exit is not dominated by bookkeeping.
- Without at least two of those, `early_stop` will lose again.

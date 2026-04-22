# Architecture Decision Records

Short, durable notes on decisions that shaped the library -- especially
the ones where the chosen option is non-obvious, or where a negative
result saves a future contributor from walking the same path twice.

## When to write an ADR

- The decision is non-obvious and the alternative looks tempting.
- A negative result generalises ("tried X, doesn't work because Y").
- The rationale is longer than a code comment but shorter than a paper.

Do **not** write an ADR to describe obvious choices, to document what
the code already says, or to narrate a release ("we did X in v0.10.3").
Release deltas belong in [`CHANGELOG.md`](../../CHANGELOG.md).

## Format

Short Michael-Nygard-style records:

```
# ADR NNNN: Title

Date: YYYY-MM-DD
Status: accepted | superseded by NNNN | deprecated

## Context
What's the situation and the constraints that force a decision?

## Decision
What was chosen?

## Consequences
What falls out of it -- positive, negative, and neutral.  Cite numbers
when they exist.
```

Keep each ADR under 150 lines.  When a decision is later reversed,
mark the old ADR "superseded by 0042" and write a new one rather than
editing the original.

## Index

| ADR | Title | Status |
|---:|---|---|
| [0001](0001-rht-off-by-default-in-pq.md) | RHT off by default in PQ indexes | accepted |
| [0002](0002-early-stop-for-ivf-probing-parked.md) | `early_stop` for IVF probing parked | accepted (negative) |
| [0003](0003-opq-p-over-opq-np.md) | OPQ-P over OPQ-NP | accepted |
| [0004](0004-cluster-contiguous-column-major-codes.md) | Cluster-contiguous column-major codes | accepted |
| [0005](0005-memory-budget-discipline.md) | Memory-budget discipline in fit and load | accepted |

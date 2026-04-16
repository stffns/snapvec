"""Microbench: IVF-PQ gather phase vectorisation.

The gather loop copies codes from probed clusters into a contiguous
(M, total) buffer, builds row_idx, and initialises scores with
coarse_dot offsets. Currently a Python loop over nprobe clusters.

Tests whether np.concatenate / fancy indexing can beat the loop.
"""

import timeit
import numpy as np

# ── Parameters matching real IVF-PQ search ──────────────────────────
M = 192
N = 20_000
NLIST = 64
NPROBE = 32
REPEATS = 200

rng = np.random.default_rng(42)

# Simulate cluster-contiguous code storage
cluster_sizes = rng.multinomial(N, np.ones(NLIST) / NLIST)
offsets = np.zeros(NLIST + 1, dtype=np.int64)
offsets[1:] = np.cumsum(cluster_sizes)
codes = rng.integers(0, 256, size=(M, N), dtype=np.uint8)
coarse_dot = rng.standard_normal(NLIST).astype(np.float32)

# Pick nprobe clusters (simulate top-nprobe selection)
probe = rng.choice(NLIST, size=NPROBE, replace=False).astype(np.int64)
starts = offsets[probe]
ends = offsets[probe + 1]
counts = ends - starts
total = int(counts.sum())

print(f"N={N:,}, nlist={NLIST}, nprobe={NPROBE}, total_candidates={total:,}, M={M}")
print()


# ── Baseline: current Python loop ──────────────────────────────────

def gather_loop():
    cat = np.empty((M, total), dtype=np.uint8)
    row_idx = np.empty(total, dtype=np.int64)
    scores = np.empty(total, dtype=np.float32)
    cursor = 0
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        cat[:, cursor:cursor + n_c] = codes[:, s:e]
        row_idx[cursor:cursor + n_c] = np.arange(s, e, dtype=np.int64)
        scores[cursor:cursor + n_c] = coarse_dot[c]
        cursor += n_c
    return cat, row_idx, scores


# ── Strategy 1: np.concatenate of pre-sliced arrays ────────────────

def gather_concatenate():
    slices_codes = [codes[:, s:e] for s, e in zip(starts.tolist(), ends.tolist()) if e > s]
    slices_idx = [np.arange(s, e, dtype=np.int64) for s, e in zip(starts.tolist(), ends.tolist()) if e > s]

    cat = np.concatenate(slices_codes, axis=1) if slices_codes else np.empty((M, 0), dtype=np.uint8)
    row_idx = np.concatenate(slices_idx) if slices_idx else np.empty(0, dtype=np.int64)

    # Build scores via repeat of coarse_dot values
    score_parts = []
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c > 0:
            score_parts.append(np.full(n_c, coarse_dot[c], dtype=np.float32))
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float32)
    return cat, row_idx, scores


# ── Strategy 2: build flat index array, single fancy-index gather ──

def gather_fancy_index():
    # Build the flat index array for all probed clusters at once
    idx_parts = [np.arange(s, e, dtype=np.int64) for s, e in zip(starts.tolist(), ends.tolist()) if e > s]
    row_idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)

    cat = codes[:, row_idx]  # single fancy-index gather on (M, N)

    # Scores: repeat coarse_dot[c] for each cluster's count
    nonzero = counts > 0
    scores = np.repeat(coarse_dot[probe[nonzero]], counts[nonzero])
    return cat, row_idx, scores


# ── Strategy 3: fully vectorised with np.repeat ────────────────────

def gather_repeat():
    # row_idx via concatenate of aranges (hard to avoid the loop for aranges)
    # But we can build it from offsets + a flat counter
    nonzero = counts > 0
    nz_starts = starts[nonzero]
    nz_counts = counts[nonzero]

    # Build row_idx: for each cluster, arange(start, end)
    # Vectorised: cumsum trick
    total_nz = int(nz_counts.sum())
    row_idx = np.empty(total_nz, dtype=np.int64)
    cursor = 0
    for s, n_c in zip(nz_starts.tolist(), nz_counts.tolist()):
        row_idx[cursor:cursor + n_c] = np.arange(s, s + n_c, dtype=np.int64)
        cursor += n_c

    cat = codes[:, row_idx]
    scores = np.repeat(coarse_dot[probe[nonzero]], nz_counts)
    return cat, row_idx, scores


# ── Strategy 4: fully vectorised row_idx via cumsum ────────────────

def gather_cumsum():
    nonzero = counts > 0
    nz_starts = starts[nonzero]
    nz_counts = counts[nonzero]
    total_nz = int(nz_counts.sum())

    # Build row_idx without Python loop using cumsum trick:
    # Place start values at cluster boundaries, fill with 1s, cumsum
    row_idx = np.ones(total_nz, dtype=np.int64)
    boundaries = np.zeros(total_nz, dtype=np.int64)
    cum_counts = np.concatenate([[0], np.cumsum(nz_counts[:-1])])
    row_idx[cum_counts] = nz_starts
    # At each boundary after the first, subtract the previous end
    # to reset the running sum
    if len(cum_counts) > 1:
        prev_ends = nz_starts[:-1] + nz_counts[:-1]
        row_idx[cum_counts[1:]] -= prev_ends - 1
    row_idx = np.cumsum(row_idx)

    cat = codes[:, row_idx]
    scores = np.repeat(coarse_dot[probe[nonzero]], nz_counts)
    return cat, row_idx, scores


# ── Validate correctness ───────────────────────────────────────────

ref_cat, ref_idx, ref_scores = gather_loop()
for name, fn in [("concatenate", gather_concatenate),
                 ("fancy_index", gather_fancy_index),
                 ("repeat", gather_repeat),
                 ("cumsum", gather_cumsum)]:
    cat, idx, scores = fn()
    assert np.array_equal(ref_cat, cat), f"{name}: cat mismatch"
    assert np.array_equal(ref_idx, idx), f"{name}: row_idx mismatch"
    assert np.allclose(ref_scores, scores, atol=1e-6), f"{name}: scores mismatch"

print("All strategies validated correct.")
print()

# ── Benchmark ──────────────────────────────────────────────────────

print("=" * 65)
print(f"GATHER PHASE  (nprobe={NPROBE}, total={total:,}, M={M})")
print("=" * 65)

baseline_us = None
for name, fn in [("loop (baseline)", gather_loop),
                 ("concatenate", gather_concatenate),
                 ("fancy_index", gather_fancy_index),
                 ("repeat (loop row_idx)", gather_repeat),
                 ("cumsum (no loop)", gather_cumsum)]:
    t = timeit.timeit(fn, number=REPEATS) / REPEATS * 1e6
    if baseline_us is None:
        baseline_us = t
    speedup = baseline_us / t
    print(f"  {name:30s}  {t:8.1f} us  {speedup:6.2f}x")

# Also break down the baseline into sub-operations
print()
print("--- Baseline breakdown ---")

def gather_only_codes():
    cat = np.empty((M, total), dtype=np.uint8)
    cursor = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        cat[:, cursor:cursor + n_c] = codes[:, s:e]
        cursor += n_c
    return cat

def gather_only_rowidx():
    row_idx = np.empty(total, dtype=np.int64)
    cursor = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        row_idx[cursor:cursor + n_c] = np.arange(s, e, dtype=np.int64)
        cursor += n_c
    return row_idx

def gather_only_scores():
    scores = np.empty(total, dtype=np.float32)
    cursor = 0
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        scores[cursor:cursor + n_c] = coarse_dot[c]
        cursor += n_c
    return scores

for name, fn in [("codes copy", gather_only_codes),
                 ("row_idx arange", gather_only_rowidx),
                 ("scores fill", gather_only_scores)]:
    t = timeit.timeit(fn, number=REPEATS) / REPEATS * 1e6
    pct = t / baseline_us * 100
    print(f"  {name:30s}  {t:8.1f} us  ({pct:5.1f}%)")

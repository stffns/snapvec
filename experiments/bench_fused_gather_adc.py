"""Bench: fused gather+ADC (skip code copy, read _codes directly)."""

import timeit
import numpy as np
import numba as nb
from snapvec._fast import _adc_colmajor_numba_par

rng = np.random.default_rng(42)
M, K, N, NLIST, NPROBE = 192, 256, 20_000, 64, 32

cluster_sizes = rng.multinomial(N, np.ones(NLIST) / NLIST)
offsets_arr = np.zeros(NLIST + 1, dtype=np.int64)
offsets_arr[1:] = np.cumsum(cluster_sizes)
codes = rng.integers(0, K, size=(M, N), dtype=np.uint8)
coarse_dot = rng.standard_normal(NLIST).astype(np.float32)
lut = rng.standard_normal((M, K)).astype(np.float32)

probe = rng.choice(NLIST, size=NPROBE, replace=False).astype(np.int64)
starts = offsets_arr[probe]
ends = offsets_arr[probe + 1]
counts = ends - starts
total = int(counts.sum())


@nb.njit(boundscheck=False, parallel=True)
def _fused_direct_adc(all_codes, row_idx, coarse_offsets, lut, scores):
    M = all_codes.shape[0]
    n = len(row_idx)
    for i in nb.prange(n):
        r = row_idx[i]
        acc = coarse_offsets[i]
        for j in range(M):
            acc += lut[j, all_codes[j, r]]
        scores[i] = acc


def build_row_idx_and_offsets():
    row_idx = np.empty(total, dtype=np.int64)
    coarse_offsets = np.empty(total, dtype=np.float32)
    cursor = 0
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        row_idx[cursor:cursor + n_c] = np.arange(s, e, dtype=np.int64)
        coarse_offsets[cursor:cursor + n_c] = coarse_dot[c]
        cursor += n_c
    return row_idx, coarse_offsets


# Warm up JIT
row_idx, coarse_offsets = build_row_idx_and_offsets()
scores_w = np.zeros(total, dtype=np.float32)
_fused_direct_adc(codes, row_idx, coarse_offsets, lut, scores_w)


def baseline():
    cat = np.empty((M, total), dtype=np.uint8)
    ri = np.empty(total, dtype=np.int64)
    sc = np.empty(total, dtype=np.float32)
    cursor = 0
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        cat[:, cursor:cursor + n_c] = codes[:, s:e]
        ri[cursor:cursor + n_c] = np.arange(s, e, dtype=np.int64)
        sc[cursor:cursor + n_c] = coarse_dot[c]
        cursor += n_c
    _adc_colmajor_numba_par(lut, cat, sc)
    return sc, ri


def fused():
    ri, co = build_row_idx_and_offsets()
    sc = np.empty(total, dtype=np.float32)
    _fused_direct_adc(codes, ri, co, lut, sc)
    return sc, ri


# Validate
ref_sc, ref_ri = baseline()
fused_sc, fused_ri = fused()
assert np.array_equal(ref_ri, fused_ri), "row_idx mismatch"
assert np.allclose(ref_sc, fused_sc, atol=1e-3), "scores mismatch"

reps = 200
t_base = timeit.timeit(baseline, number=reps) / reps * 1e6
t_fused = timeit.timeit(fused, number=reps) / reps * 1e6

print(f"nprobe={NPROBE}, total={total:,}, M={M}, N={N:,}")
print(f"  gather + parallel ADC:  {t_base:7.0f} us")
print(f"  fused direct ADC:       {t_fused:7.0f} us  ({t_base/t_fused:.2f}x)")

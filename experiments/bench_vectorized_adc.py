"""Microbench: vectorised ADC via np.take_along_axis vs per-subspace loop.

Tests two independent optimisations:
  1. LUT construction: M separate matmuls vs one batched np.matmul
  2. ADC scoring: M-iteration loop vs single take_along_axis + sum

Covers both IVF-PQ layout (codes shape M, n -- column-major) and
PQ layout (codes shape n, M -- row-major).
"""

import timeit
import numpy as np

# ── Parameters matching real workloads ──────────────────────────────
M = 192           # subspaces
K = 256           # codewords per subspace
d_sub = 2         # dim / M  (384 / 192)
N_CANDIDATES = [500, 1_000, 3_500, 10_000, 50_000]
REPEATS = 200

rng = np.random.default_rng(42)

# Shared fixtures
codebooks = rng.standard_normal((M, K, d_sub)).astype(np.float32)
q_pre = rng.standard_normal(M * d_sub).astype(np.float32)
q_split = q_pre.reshape(M, d_sub)


# ====================================================================
# 1. LUT construction: loop vs batched matmul
# ====================================================================

def lut_loop():
    lut = np.empty((M, K), dtype=np.float32)
    for j in range(M):
        qj = q_pre[j * d_sub : (j + 1) * d_sub]
        lut[j] = codebooks[j] @ qj
    return lut


def lut_batched():
    return np.matmul(codebooks, q_split[:, :, np.newaxis]).squeeze(-1)


def lut_einsum():
    return np.einsum("mks,ms->mk", codebooks, q_split)


# Validate correctness
ref = lut_loop()
assert np.allclose(ref, lut_batched(), atol=1e-5), "batched matmul mismatch"
assert np.allclose(ref, lut_einsum(), atol=1e-5), "einsum mismatch"

print("=" * 70)
print("LUT CONSTRUCTION  (M={}, K={}, d_sub={})".format(M, K, d_sub))
print("=" * 70)

for name, fn in [("loop (baseline)", lut_loop),
                 ("batched matmul", lut_batched),
                 ("einsum", lut_einsum)]:
    t = timeit.timeit(fn, number=REPEATS) / REPEATS * 1e6
    print(f"  {name:25s}  {t:8.1f} us")


# ====================================================================
# 2. ADC scoring: loop vs take_along_axis
# ====================================================================

lut = ref  # (M, K)

print()
print("=" * 70)
print("ADC SCORING  (M={}, K={})".format(M, K))
print("=" * 70)


for n_cand in N_CANDIDATES:
    # IVF-PQ layout: codes (M, n) -- column-major
    codes_col = rng.integers(0, K, size=(M, n_cand), dtype=np.uint8)

    # PQ layout: codes (n, M) -- row-major
    codes_row = codes_col.T.copy()

    # -- Baselines (current code) --

    def adc_loop_colmajor():
        scores = np.zeros(n_cand, dtype=np.float32)
        for j in range(M):
            scores += lut[j][codes_col[j]]
        return scores

    def adc_loop_rowmajor():
        scores = np.zeros(n_cand, dtype=np.float32)
        for j in range(M):
            scores += lut[j][codes_row[:, j]]
        return scores

    # -- Vectorised candidates --

    def adc_take_colmajor():
        gathered = np.take_along_axis(lut, codes_col, axis=1)  # (M, n_cand)
        return gathered.sum(axis=0)

    def adc_take_rowmajor():
        # Row-major (n, M) -> transpose to (M, n) for take_along_axis
        gathered = np.take_along_axis(lut, codes_col, axis=1)  # (M, n)
        return gathered.sum(axis=0)

    def adc_fancy_colmajor():
        j_idx = np.arange(M)[:, np.newaxis]  # (M, 1)
        return lut[j_idx, codes_col].sum(axis=0)

    def adc_take_colmajor_inplace():
        gathered = np.take_along_axis(lut, codes_col, axis=1)
        return np.add.reduce(gathered, axis=0)

    # Chunked: process CHUNK subspaces at a time to balance dispatch
    # count vs intermediate allocation size.
    def make_chunked(chunk_size):
        def adc_chunked():
            scores = np.zeros(n_cand, dtype=np.float32)
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                g = np.take_along_axis(lut[start:end], codes_col[start:end], axis=1)
                scores += g.sum(axis=0)
            return scores
        return adc_chunked

    adc_chunked_8 = make_chunked(8)
    adc_chunked_16 = make_chunked(16)
    adc_chunked_32 = make_chunked(32)
    adc_chunked_64 = make_chunked(64)

    # Flat indexing: single fancy index into ravelled LUT
    offsets = (np.arange(M, dtype=np.int32) * K)[:, np.newaxis]  # (M, 1)
    def adc_flat():
        flat_idx = codes_col.astype(np.int32) + offsets  # (M, n_cand)
        return lut.ravel()[flat_idx].sum(axis=0)

    # Pre-computed flat codes (simulates storing int32 flat codes at build time)
    codes_flat_pre = codes_col.astype(np.int32) + offsets  # precomputed
    def adc_flat_precomputed():
        return lut.ravel()[codes_flat_pre].sum(axis=0)

    # Flat + ravel with .ravel() on gather output for contiguous sum
    def adc_flat_pre_ravel():
        gathered = lut.ravel()[codes_flat_pre.ravel()]  # (M * n_cand,) contiguous
        return gathered.reshape(M, n_cand).sum(axis=0)

    # Sparse matvec: CSR @ lut.ravel()
    try:
        from scipy import sparse
        rows = np.repeat(np.arange(n_cand, dtype=np.int32), M)
        cols = codes_flat_pre.T.ravel()  # (n_cand * M,)
        data = np.ones(n_cand * M, dtype=np.float32)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n_cand, M * K))
        def adc_sparse():
            return A @ lut.ravel()
        has_sparse = True
    except ImportError:
        has_sparse = False

    # Validate
    ref_scores = adc_loop_colmajor()
    for name, fn in [("take_along_axis (col)", adc_take_colmajor),
                     ("fancy indexing (col)", adc_fancy_colmajor),
                     ("add.reduce (col)", adc_take_colmajor_inplace),
                     ("chunked-8", adc_chunked_8),
                     ("chunked-16", adc_chunked_16),
                     ("chunked-32", adc_chunked_32),
                     ("chunked-64", adc_chunked_64),
                     ("flat indexing", adc_flat),
                     ("flat precomputed", adc_flat_precomputed),
                     ("flat pre ravel", adc_flat_pre_ravel)]:
        result = fn()
        assert np.allclose(ref_scores, result, atol=1e-4), f"{name} mismatch"
    if has_sparse:
        result = adc_sparse()
        assert np.allclose(ref_scores, result, atol=1e-4), "sparse mismatch"

    reps = max(20, REPEATS // (n_cand // 500))

    print(f"\n  n_cand = {n_cand:,}")
    print(f"  {'method':35s}  {'us':>8s}  {'vs baseline':>11s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*11}")

    baseline_us = None
    methods = [("loop (baseline, col-major)", adc_loop_colmajor),
               ("loop (baseline, row-major)", adc_loop_rowmajor),
               ("take_along_axis (col)", adc_take_colmajor),
               ("fancy indexing (col)", adc_fancy_colmajor),
               ("add.reduce (col)", adc_take_colmajor_inplace),
               ("chunked-8", adc_chunked_8),
               ("chunked-16", adc_chunked_16),
               ("chunked-32", adc_chunked_32),
               ("chunked-64", adc_chunked_64),
               ("flat indexing", adc_flat),
               ("flat precomputed", adc_flat_precomputed),
               ("flat pre ravel", adc_flat_pre_ravel)]
    if has_sparse:
        methods.append(("sparse CSR @ lut", adc_sparse))
    for name, fn in methods:
        t = timeit.timeit(fn, number=reps) / reps * 1e6
        if baseline_us is None:
            baseline_us = t
        speedup = baseline_us / t
        print(f"  {name:35s}  {t:8.1f}  {speedup:10.2f}x")

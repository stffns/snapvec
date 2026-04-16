"""End-to-end search pipeline profile for IVFPQSnapIndex and PQSnapIndex.

Breaks down wall-clock time per stage to identify the new bottleneck
distribution after the batched-matmul LUT optimisation (v0.8.1).
"""

import time
import numpy as np
from snapvec._ivfpq import IVFPQSnapIndex
from snapvec._pq import PQSnapIndex
from snapvec._kmeans import probe_scores_l2_monotone

rng = np.random.default_rng(42)

# ── Realistic-ish corpus ────────────────────────────────────────────
DIM = 384
N = 20_000
M = 192
K = 256
NLIST = 64
NPROBE = 32
QUERIES = 50

corpus = rng.standard_normal((N, DIM)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
queries = rng.standard_normal((QUERIES, DIM)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
ids = list(range(N))


# ====================================================================
# 1. IVFPQSnapIndex stage-by-stage profile
# ====================================================================

print("Building IVFPQSnapIndex (N={:,}, dim={}, M={}, nlist={})...".format(N, DIM, M, NLIST))
ivf = IVFPQSnapIndex(dim=DIM, M=M, K=K, nlist=NLIST, normalized=True)
ivf.fit(corpus[:2000])
ivf.add_batch(ids, corpus)

# Warm up caches
for q in queries[:3]:
    ivf.search(q, k=10, nprobe=NPROBE)

# Profile each stage manually
timings = {
    "preprocess": [],
    "coarse_probe": [],
    "lut_build": [],
    "gather": [],
    "adc_loop": [],
    "norms": [],
    "topk": [],
    "total": [],
}

for q in queries:
    q_f = np.asarray(q, dtype=np.float32)

    t0 = time.perf_counter()
    # preprocess
    q_pre = ivf._preprocess_single(q_f)
    t1 = time.perf_counter()

    # coarse probe
    probe_ranking = probe_scores_l2_monotone(ivf._coarse, q_pre)
    coarse_dot = ivf._coarse @ q_pre
    probe = ivf._probe_topn(probe_ranking, NPROBE, None)
    t2 = time.perf_counter()

    # LUT build (batched matmul)
    q_split = q_pre.reshape(ivf.M, ivf._d_sub, 1)
    lut = np.matmul(ivf._codebooks, q_split).squeeze(-1)
    t3 = time.perf_counter()

    # Gather codes from probed clusters
    starts = ivf._offsets[probe]
    ends = ivf._offsets[probe + 1]
    total_cand = int((ends - starts).sum())
    cat = np.empty((ivf.M, total_cand), dtype=np.uint8)
    row_idx = np.empty(total_cand, dtype=np.int64)
    scores = np.empty(total_cand, dtype=np.float32)
    cursor = 0
    for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
        n_c = e - s
        if n_c == 0:
            continue
        cat[:, cursor:cursor + n_c] = ivf._codes[:, s:e]
        row_idx[cursor:cursor + n_c] = np.arange(s, e, dtype=np.int64)
        scores[cursor:cursor + n_c] = coarse_dot[c]
        cursor += n_c
    t4 = time.perf_counter()

    # ADC scoring loop
    for j in range(ivf.M):
        scores += lut[j][cat[j]]
    t5 = time.perf_counter()

    # Norms (skip for normalized=True, but measure the branch)
    t6 = t5  # normalized=True, no norms step

    # Top-k
    k = 10
    k_eff = min(k, total_cand)
    top = np.argpartition(-scores, k_eff - 1)[:k_eff]
    top = top[np.argsort(-scores[top])]
    t7 = time.perf_counter()

    timings["preprocess"].append(t1 - t0)
    timings["coarse_probe"].append(t2 - t1)
    timings["lut_build"].append(t3 - t2)
    timings["gather"].append(t4 - t3)
    timings["adc_loop"].append(t5 - t4)
    timings["topk"].append(t7 - t6)
    timings["total"].append(t7 - t0)

print()
print("=" * 70)
print("IVFPQSnapIndex SEARCH PROFILE  (nprobe={}, ~{:,} candidates)".format(
    NPROBE, total_cand))
print("=" * 70)

total_us = np.median(timings["total"]) * 1e6
for stage in ["preprocess", "coarse_probe", "lut_build", "gather", "adc_loop", "topk"]:
    med = np.median(timings[stage]) * 1e6
    pct = med / total_us * 100
    print(f"  {stage:20s}  {med:8.1f} us  ({pct:5.1f}%)")
print(f"  {'TOTAL':20s}  {total_us:8.1f} us")


# ====================================================================
# 2. PQSnapIndex stage-by-stage profile
# ====================================================================

print()
print("Building PQSnapIndex (N={:,}, dim={}, M={})...".format(N, DIM, M))
pq = PQSnapIndex(dim=DIM, M=M, K=K, normalized=True)
pq.fit(corpus[:2000])
pq.add_batch(ids, corpus)

# Warm up
for q in queries[:3]:
    pq.search(q, k=10)

pq_timings = {
    "preprocess": [],
    "lut_build": [],
    "adc_loop": [],
    "topk": [],
    "total": [],
}

for q in queries:
    q_f = np.asarray(q, dtype=np.float32)

    t0 = time.perf_counter()
    q_pre = pq._preprocess_single(q_f)
    t1 = time.perf_counter()

    q_split = q_pre.reshape(pq.M, pq._d_sub, 1)
    lut = np.matmul(pq._codebooks, q_split).squeeze(-1)
    t2 = time.perf_counter()

    scores = np.zeros(len(pq._codes), dtype=np.float32)
    for j in range(pq.M):
        scores += lut[j][pq._codes[:, j]]
    t3 = time.perf_counter()

    k_eff = min(10, len(scores))
    top = np.argpartition(-scores, k_eff - 1)[:k_eff]
    top = top[np.argsort(-scores[top])]
    t4 = time.perf_counter()

    pq_timings["preprocess"].append(t1 - t0)
    pq_timings["lut_build"].append(t2 - t1)
    pq_timings["adc_loop"].append(t3 - t2)
    pq_timings["topk"].append(t4 - t3)
    pq_timings["total"].append(t4 - t0)

print()
print("=" * 70)
print("PQSnapIndex SEARCH PROFILE  (full scan, N={:,})".format(N))
print("=" * 70)

total_us = np.median(pq_timings["total"]) * 1e6
for stage in ["preprocess", "lut_build", "adc_loop", "topk"]:
    med = np.median(pq_timings[stage]) * 1e6
    pct = med / total_us * 100
    print(f"  {stage:20s}  {med:8.1f} us  ({pct:5.1f}%)")
print(f"  {'TOTAL':20s}  {total_us:8.1f} us")


# ====================================================================
# 3. PQSnapIndex: row-major vs column-major ADC comparison
# ====================================================================

print()
print("=" * 70)
print("PQ ADC: ROW-MAJOR (n,M) vs COLUMN-MAJOR (M,n)")
print("=" * 70)

import timeit

codes_row = pq._codes                          # (n, M) -- current layout
codes_col = pq._codes.T.copy()                 # (M, n) -- IVF-PQ layout

# Build LUT once for fair comparison
q_pre = pq._preprocess_single(queries[0].astype(np.float32))
q_split = q_pre.reshape(pq.M, pq._d_sub, 1)
lut = np.matmul(pq._codebooks, q_split).squeeze(-1)

def adc_row():
    scores = np.zeros(N, dtype=np.float32)
    for j in range(M):
        scores += lut[j][codes_row[:, j]]
    return scores

def adc_col():
    scores = np.zeros(N, dtype=np.float32)
    for j in range(M):
        scores += lut[j][codes_col[j]]
    return scores

# Validate equivalence
assert np.allclose(adc_row(), adc_col(), atol=1e-5)

reps = 50
t_row = timeit.timeit(adc_row, number=reps) / reps * 1e6
t_col = timeit.timeit(adc_col, number=reps) / reps * 1e6
print(f"  row-major (n,M)  {t_row:8.1f} us  (current PQ layout)")
print(f"  col-major (M,n)  {t_col:8.1f} us  ({t_row/t_col:.2f}x speedup)")

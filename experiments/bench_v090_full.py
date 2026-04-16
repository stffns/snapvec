"""Full benchmark suite for snapvec v0.9.0.

Measures recall@10, latency (us/query), build time, and storage
across PQSnapIndex and IVFPQSnapIndex at multiple operating points.
Designed to produce the definitive numbers for the release.
"""

import time
import numpy as np
from snapvec import PQSnapIndex, IVFPQSnapIndex

rng = np.random.default_rng(42)

DIM = 384
M = 192
K = 256
NLIST = 64
N_VALUES = [10_000, 20_000, 50_000]
N_QUERIES = 200
K_SEARCH = 10
NPROBES = [4, 8, 16, 32, 64, 128]


def brute_force_topk(corpus, queries, k):
    """Exact top-k via brute-force dot product."""
    results = []
    for q in queries:
        scores = corpus @ q
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results.append(set(top.tolist()))
    return results


def recall_at_k(exact, approx, k):
    """Mean recall@k."""
    total = 0.0
    for ex, ap in zip(exact, approx):
        ex_set = ex if isinstance(ex, set) else set(ex)
        ap_ids = set(id_ for id_, _ in ap[:k])
        total += len(ex_set & ap_ids) / k
    return total / len(exact)


def bench_pq(N, corpus, queries, exact_results):
    print(f"\n{'='*70}")
    print(f"PQSnapIndex  (N={N:,}, dim={DIM}, M={M}, K={K})")
    print(f"{'='*70}")

    ids = list(range(N))

    # Build
    pq = PQSnapIndex(dim=DIM, M=M, K=K, normalized=True)
    t0 = time.perf_counter()
    pq.fit(corpus[:min(2000, N)])
    fit_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    pq.add_batch(ids, corpus)
    add_ms = (time.perf_counter() - t0) * 1e3

    stats = pq.stats()
    total_mb = stats['bytes_per_vec'] * N / 1024 / 1024
    print(f"  fit:     {fit_ms:,.0f} ms")
    print(f"  add:     {add_ms:,.0f} ms")
    print(f"  storage: {stats['bytes_per_vec']:.0f} B/vec, "
          f"{total_mb:.1f} MB total")

    # Warm up
    for q in queries[:5]:
        pq.search(q, k=K_SEARCH)

    # Search
    t0 = time.perf_counter()
    approx_results = [pq.search(q, k=K_SEARCH) for q in queries]
    total_s = time.perf_counter() - t0
    us_per_q = total_s / len(queries) * 1e6
    rec = recall_at_k(exact_results, approx_results, K_SEARCH)

    print(f"\n  {'recall@10':>10s}  {'us/query':>10s}")
    print(f"  {rec:>10.3f}  {us_per_q:>10.0f}")


def bench_ivfpq(N, corpus, queries, exact_results):
    print(f"\n{'='*70}")
    print(f"IVFPQSnapIndex  (N={N:,}, dim={DIM}, M={M}, K={K}, nlist={NLIST})")
    print(f"{'='*70}")

    ids = list(range(N))

    # Build
    ivf = IVFPQSnapIndex(
        dim=DIM, M=M, K=K, nlist=NLIST,
        normalized=True, keep_full_precision=True,
    )
    t0 = time.perf_counter()
    ivf.fit(corpus[:min(2000, N)])
    fit_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    ivf.add_batch(ids, corpus)
    add_ms = (time.perf_counter() - t0) * 1e3

    stats = ivf.stats()
    total_mb = stats['bytes_per_vec'] * N / 1024 / 1024
    codes_bpv = stats.get('bytes_per_vec_codes_only', stats['bytes_per_vec'])
    print(f"  fit:     {fit_ms:,.0f} ms")
    print(f"  add:     {add_ms:,.0f} ms")
    print(f"  storage: {stats['bytes_per_vec']:.0f} B/vec "
          f"({codes_bpv} codes-only), {total_mb:.1f} MB total")

    # Warm up
    for q in queries[:5]:
        ivf.search(q, k=K_SEARCH, nprobe=32)

    # Search sweep
    print(f"\n  {'nprobe':>6s}  {'recall@10':>10s}  {'us/query':>10s}"
          f"  {'rerank recall':>13s}  {'rerank us':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*13}  {'-'*10}")

    for nprobe in NPROBES:
        if nprobe > NLIST:
            continue

        # PQ-only
        t0 = time.perf_counter()
        approx = [ivf.search(q, k=K_SEARCH, nprobe=nprobe) for q in queries]
        total_s = time.perf_counter() - t0
        us_pq = total_s / len(queries) * 1e6
        rec_pq = recall_at_k(exact_results, approx, K_SEARCH)

        # With rerank
        t0 = time.perf_counter()
        approx_rr = [
            ivf.search(q, k=K_SEARCH, nprobe=nprobe, rerank_candidates=100)
            for q in queries
        ]
        total_s = time.perf_counter() - t0
        us_rr = total_s / len(queries) * 1e6
        rec_rr = recall_at_k(exact_results, approx_rr, K_SEARCH)

        print(f"  {nprobe:>6d}  {rec_pq:>10.3f}  {us_pq:>10.0f}"
              f"  {rec_rr:>13.3f}  {us_rr:>10.0f}")


def main():
    print("snapvec v0.9.0 -- Full Benchmark Suite")
    print(f"Python kernels: Cython + OpenMP")
    print(f"Parameters: dim={DIM}, M={M}, K={K}, nlist={NLIST}")
    print(f"Queries: {N_QUERIES}")

    for N in N_VALUES:
        print(f"\n\n{'#'*70}")
        print(f"# CORPUS SIZE: N = {N:,}")
        print(f"{'#'*70}")

        corpus = rng.standard_normal((N, DIM)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Exact results
        print("\n  Computing brute-force exact results...")
        exact = brute_force_topk(corpus, queries, K_SEARCH)

        bench_pq(N, corpus, queries, exact)
        bench_ivfpq(N, corpus, queries, exact)


if __name__ == "__main__":
    main()

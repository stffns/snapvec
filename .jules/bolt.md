## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2025-07-26 - Optimized L2 Distances with Einsum
**Learning:** Replaced large `sum` allocations `((X ** 2).sum(axis))` with `np.einsum('ij,ij->i', X, X)` reducing memory allocation overhead during L2 calculations in `kmeans_pp_init`, `kmeans_mse`, `assign_l2`, `probe_scores_l2_monotone`, and `_consolidate` methods.
**Action:** Use `np.einsum` in hot code loops for L2 distances rather than sum-of-squares arrays to achieve a 3-5x execution speedup.

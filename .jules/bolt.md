## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2026-06-11 - Fast 1D vector norm in pure NumPy
**Learning:** While `np.sqrt(np.einsum(...))` is fast for 2D batched arrays, calculating the L2 norm of 1D NumPy arrays (e.g., single queries) is significantly faster (~1.6x speedup) using `np.sqrt(np.vdot(q, q))` than using `np.linalg.norm(q)`. It correctly handles complex conjugation and executes faster for float32 arrays by bypassing Python-level overhead and dimension checking inherent in `np.linalg.norm`.
**Action:** Use `np.sqrt(np.vdot(q, q))` instead of `np.linalg.norm(q)` for calculating L2 norms of single 1D arrays to maximize performance.

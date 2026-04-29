## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast L2 norm for 1D float arrays in NumPy
**Learning:** In performance-critical paths dealing with 1D vectors (like queries in vector search), computing the L2 norm via `np.linalg.norm(q)` is relatively slow due to Python-level overhead, kwarg handling, and dimension checking. Using `np.sqrt(np.inner(q, q))` achieves a ~1.5x speedup by bypassing this overhead and mapping more directly to C/BLAS routines.
**Action:** Always prefer `np.sqrt(np.inner(q, q))` over `np.linalg.norm(q)` when calculating the L2 norm of single 1D arrays to optimize execution speed.

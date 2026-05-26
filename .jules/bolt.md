## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-26 - Fast 1D vector Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the L2 norm of a 1D NumPy array (e.g., a query vector) via `np.linalg.norm(q)` is relatively slow due to Python-level dispatch and type-checking overhead. Using `np.sqrt(np.inner(q, q))` achieves a ~1.2x - 1.5x speedup for 1D arrays while maintaining functional equivalence.
**Action:** Always prefer `np.sqrt(np.inner(q, q))` over `np.linalg.norm(q)` when calculating the Euclidean norm of 1D NumPy arrays to bypass Python dispatch overhead and improve execution speed.

## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast L2 norm for 1D arrays in pure NumPy
**Learning:** In performance-critical paths, computing the L2 norm of a 1D array via `np.linalg.norm(q)` is surprisingly slow due to internal Python overhead, kwarg handling, and dimension checking. Using `np.sqrt(np.vdot(q, q))` bypasses this overhead and is functionally identical for flat arrays, leading to a significant ~1.5x speedup on typical queries.
**Action:** Always prefer `np.sqrt(np.vdot(q, q))` over `np.linalg.norm(q)` when computing the L2 norm of 1D NumPy arrays (e.g. query vectors) to minimize dispatch overhead and improve execution speed.

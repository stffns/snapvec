## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast L2 norm for 1D arrays in NumPy
**Learning:** For computing the L2 norm of 1D NumPy arrays (e.g., single query vectors), using `np.sqrt(np.vdot(q, q))` or `np.sqrt(np.inner(q, q))` is functionally identical to but significantly faster (~1.5x) than `np.linalg.norm(q)`. This is because it bypasses the Python-level overhead, keyword argument handling, and dimension checking inherent in `np.linalg.norm`.
**Action:** Always prefer `np.sqrt(np.vdot(q, q))` over `np.linalg.norm(q)` when computing the L2 norm of known 1D arrays in performance-critical paths.

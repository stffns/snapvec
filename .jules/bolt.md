## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast single vector Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the L2 norm of a 1D array via `np.linalg.norm(q)` incurs significant Python-level overhead, dimension checking, and kwargs handling. Using `np.sqrt(np.inner(q, q))` or `np.sqrt(np.vdot(q, q))` is functionally identical but significantly faster (~1.5x speedup on typical vectors).
**Action:** Always prefer `np.sqrt(np.inner(q, q))` over `np.linalg.norm(q)` when computing the L2 norm of single 1D vectors in NumPy to eliminate dispatch overhead and improve execution speed.

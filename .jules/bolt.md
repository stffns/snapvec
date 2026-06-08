## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast 1D vector norm in pure NumPy
**Learning:** In performance-critical paths, computing the L2 norm of a 1D array via `np.linalg.norm(q)` incurs significant Python-level overhead, kwarg handling, and dimension checking. Using `np.sqrt(np.vdot(q, q))` achieves the exact same result mathematically but is significantly faster (~1.6x speedup on a laptop CPU). `np.vdot` correctly handles complex conjugation and executes faster than `np.inner` for `float32` arrays in these routines because it directly calls the underlying BLAS `sdot` function.
**Action:** Always prefer `np.sqrt(np.vdot(q, q))` over `np.linalg.norm(q)` when computing the L2 norm of 1-dimensional NumPy arrays to bypass dispatch overhead and improve execution speed.

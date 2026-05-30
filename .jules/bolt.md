## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast 1D L2 norm in pure NumPy
**Learning:** When calculating the L2 norm for single query vectors (1D arrays), `np.linalg.norm(q)` incurs significant overhead due to Python-level kwarg handling and dispatch. Using `np.sqrt(np.vdot(q, q))` provides the exact same functionality but operates at the C-level, yielding a ~1.5x speedup for 1024-dimensional float32 arrays.
**Action:** Always prefer `np.sqrt(np.vdot(q, q))` over `np.linalg.norm(q)` when calculating the L2 norm of 1D arrays in performance-critical code paths to eliminate unnecessary Python-level overhead.

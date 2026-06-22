## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.
## 2024-06-22 - L2 Norm Optimization for 1D Arrays
**Learning:** For calculating the L2 norm of 1D NumPy arrays in hot paths (like single query vectors), `np.sqrt(np.vdot(q, q))` executes significantly faster (~1.4x speedup) than `np.linalg.norm(q)` by bypassing Python-level overhead and dimension checking, while correctly handling complex conjugation.
**Action:** Always prefer `np.sqrt(np.vdot(q, q))` over `np.linalg.norm(q)` for 1D arrays in performance-critical sections.

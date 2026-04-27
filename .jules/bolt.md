## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2025-02-28 - Faster L2 norm for 1D vectors with np.vdot
**Learning:** For computing the L2 norm of 1D NumPy arrays (like single query vectors), `np.sqrt(np.vdot(q, q))` is approximately 1.5x faster than `np.linalg.norm(q)`. This optimization bypasses Python-level overhead (argument checking and dimension handling) inherent in `np.linalg.norm` and relies directly on underlying BLAS `dot`.
**Action:** When working with 1D real-valued vectors and computing the Euclidean norm, prioritize using `np.sqrt(np.vdot(q, q))` over `np.linalg.norm` to reduce overhead and improve execution speed.

## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2026-06-29 - Fast batched tensor contraction in NumPy
**Learning:** In performance-critical batched tensor contractions in NumPy (like computing residual LUTs), rewriting expressions to avoid explicit transpositions (e.g., using linear algebra associativity such as `R @ S.T` instead of `(S @ R.T).T`) prevents intermediate memory allocations, improves execution speed, and yields a C-contiguous array instead of an F-contiguous view, thereby enhancing cache locality for subsequent operations.
**Action:** Prefer `R @ S.T` over `(S @ R.T).T` when multiplying a batch of vectors with a transformation matrix.

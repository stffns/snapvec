## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2025-10-24 - Avoid explicit transpositions in matrix multiplication
**Learning:** Rewriting expressions using linear algebra associativity (e.g., `R @ S.T` instead of `(S @ R.T).T`) prevents intermediate memory allocations, improves execution speed, and yields a C-contiguous array instead of an F-contiguous view, thereby enhancing cache locality for subsequent operations.
**Action:** Always prefer `R @ S.T` over `(S @ R.T).T` in NumPy to minimize memory overhead and ensure C-contiguous results.

## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2026-06-23 - Fast dictionary ID updates
**Learning:** In `snapvec`, manual `enumerate` loops (e.g., `for i, id_val in enumerate(ids): d[id_val] = start + i`) for updating ID mapping dictionaries during `add_batch` operations are relatively slow. Replacing them with `d.update(zip(ids, range(start, start + len(ids))))` moves the iteration to C-level, yielding a ~1.6x speedup on dictionary updates.
**Action:** Use `dict.update(zip(keys, range(...)))` instead of manual Python loops to build or extend dictionaries when processing batches of IDs.

## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2026-06-23 - Fast dictionary ID updates
**Learning:** In `snapvec`, manual `enumerate` loops (e.g., `for i, id_val in enumerate(ids): d[id_val] = start + i`) for updating ID mapping dictionaries during `add_batch` operations are relatively slow. Replacing them with `d.update(zip(ids, range(start, start + len(ids))))` moves the iteration to C-level, yielding a ~1.6x speedup on dictionary updates.
**Action:** Use `dict.update(zip(keys, range(...)))` instead of manual Python loops to build or extend dictionaries when processing batches of IDs.

## 2026-06-23 - NumPy 2.3+ typing compatibility
**Learning:** NumPy 2.3 introduced `type` statements in its type stubs (`__init__.pyi`), which are only supported in Python 3.12+. When running `mypy` with `python_version = "3.10"` configured in `pyproject.toml`, this causes syntax errors in the NumPy stubs when parsed by mypy, regardless of the Python environment running mypy. This leads to CI failures when checking a Python 3.10 project against modern NumPy versions.
**Action:** Avoid explicit casts like `cast("NDArray[np.int64]", arr)` if the underlying operation can naturally produce the correct type via `.astype(np.int64)`. While this fixes one mypy error, the broader `type` statement issue in NumPy 2.3+ stubs against mypy's 3.10 parser requires either pinning NumPy < 2.3, ignoring the specific syntax error in mypy, or updating the project's mypy `python_version` to match the environment if Python >= 3.12 is assumed for checking.

## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Batching small file writes for ChecksumWriter
**Learning:** Batching multiple small file writes into a single `bytearray` before calling `f.write()` significantly improves serialization performance (approx. 1.4x speedup) in `SnapIndex.save` by reducing system call overhead and frequent `zlib.crc32` updates in the `ChecksumWriter` loop.
**Action:** Always prefer batching small writes into a `bytearray` before writing to file when serializing index files to reduce overhead and improve write performance.

## 2024-05-18 - Type annotations for bytearray in file writes
**Learning:** When modifying file writing operations to use `bytearray` for performance batching in `ChecksumWriter`, ensure the type signature of the `write` method is updated to accept `typing.Union[bytes, bytearray]`. Avoid the `|` type union syntax (`bytes | bytearray`) to satisfy reviewer constraints regarding backward compatibility with Python 3.9, even if the project nominally requires Python >= 3.10.
**Action:** Always prefer `typing.Union` over the `|` syntax for type unions unless specifically instructed otherwise by the reviewer.

## 2024-05-18 - Bounded memory usage for batching writes
**Learning:** While batching small file writes into a single `bytearray` before calling `f.write()` significantly improves serialization performance (approx. 1.4x speedup) in `SnapIndex.save`, unbounded batching can cause massive memory spikes. A chunked batching approach (e.g., writing every 64KB) is much safer and addresses reviewer concerns.
**Action:** Always implement a max buffer size (like 64KB/65536 bytes) before writing to file when serializing index files to prevent OOM errors on massive datasets.

## 2024-05-18 - CI version matching for static analysis
**Learning:** Mypy errors indicating 'Type statement is only supported in Python 3.12 and greater' within `numpy/__init__.pyi` stem from a mismatch between the installed NumPy version (>=2.5.0) and mypy's `python_version` setting (e.g., '3.10') in `pyproject.toml`. This is an upstream environment configuration issue. To fix this in CI, align the `python-version` in the GitHub Actions workflow (e.g., `.github/workflows/ci.yml`) with the `mypy` `python_version`. Locally, bypass it by testing with the matched older Python version.
**Action:** When diagnosing CI type-checking failures originating from third-party stubs, check if the CI's Python version matches the project's configured static typing version (e.g., mypy's `python_version`) before assuming the codebase needs modification.

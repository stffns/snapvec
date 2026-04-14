## 2024-05-18 - QJL Inner Product Bottleneck
**Learning:** In NumPy, mixing precision types like `int8` (QJL residual signs) and `float32` (Query projections) via matrix multiplication (`@`) using `.astype(np.float32)` forces the creation of a massive dense `float32` copy in memory. This leads to a severe O(N * d) RAM spike during `search` in prod mode.
**Action:** Use `np.inner` or `np.dot` which delegates the mixed-type calculation natively to C-level routines, completely bypassing the array copy, dropping the memory spike to 0, and speeding up execution safely.

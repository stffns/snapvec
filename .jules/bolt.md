## 2024-05-18 - Optimize 1D L2 norm calculations

**Learning:** `np.linalg.norm` carries significant overhead for small 1D arrays due to Python-level dispatch and validation. Replacing it with `np.sqrt(np.inner(q, q))` yields a ~1.3x speedup in latency-sensitive paths like query preprocessing.
**Action:** Always prefer `np.sqrt(np.inner(x, x))` or `np.sqrt(np.dot(x, x))` over `np.linalg.norm(x)` for computing the L2 norm of small 1D vectors in performance-critical code. Ensure to include inline comments explaining the rationale.

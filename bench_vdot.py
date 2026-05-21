import numpy as np
import time

q = np.random.randn(384).astype(np.float32)

def bench_linalg_norm():
    return float(np.linalg.norm(q))

def bench_inner_sqrt():
    return float(np.sqrt(np.inner(q, q)))

def bench(func, n=100000):
    start = time.time()
    for _ in range(n):
        func()
    return time.time() - start

print(f"linalg.norm: {bench(bench_linalg_norm):.4f}s")
print(f"inner+sqrt:  {bench(bench_inner_sqrt):.4f}s")

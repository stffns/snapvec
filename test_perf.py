import time
import numpy as np

np.random.seed(0)
M = 32
K = 256
d_sub = 16
N = 10000

cb = np.random.randn(M, K, d_sub).astype(np.float32)

t0 = time.time()
for _ in range(100):
    cb_norms1 = (cb ** 2).sum(2)
t1 = time.time()
print(f"Time ** 2 sum: {t1 - t0}")

t0 = time.time()
for _ in range(100):
    cb_norms2 = np.einsum('ijk,ijk->ij', cb, cb)
t1 = time.time()
print(f"Time einsum: {t1 - t0}")

print(np.allclose(cb_norms1, cb_norms2))

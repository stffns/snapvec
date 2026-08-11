import time
import numpy as np

np.random.seed(0)

# Simulate ivfpq codebooks sum(2)
M = 32
K = 256
d_sub = 24
cb = np.random.randn(M, K, d_sub).astype(np.float32)

t0 = time.time()
for _ in range(1000):
    cb_norms1 = (cb ** 2).sum(2)
t1 = time.time()
print(f"Time original sum(2) [M={M}, K={K}, d_sub={d_sub}]: {t1 - t0}")

t0 = time.time()
for _ in range(1000):
    cb_norms2 = np.einsum('ijk,ijk->ij', cb, cb)
t1 = time.time()
print(f"Time einsum sum(2) [M={M}, K={K}, d_sub={d_sub}]: {t1 - t0}")

print(np.allclose(cb_norms1, cb_norms2))

import numpy as np
import time

def assign_l2_original(X, C):
    d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return d2.argmin(1)

def assign_l2_einsum(X, C):
    d2 = np.einsum('ij,ij->i', X, X)[:, None] - 2 * X @ C.T + np.einsum('ij,ij->i', C, C)[None, :]
    return d2.argmin(1)

# Dummy data
N, D = 100000, 128
K = 256
np.random.seed(0)
X = np.random.randn(N, D).astype(np.float32)
C = np.random.randn(K, D).astype(np.float32)

# Warmup
assign_l2_original(X[:10], C[:10])
assign_l2_einsum(X[:10], C[:10])

t0 = time.time()
res1 = assign_l2_original(X, C)
t1 = time.time()
print(f"Original: {t1 - t0:.4f}s")

t0 = time.time()
res2 = assign_l2_einsum(X, C)
t1 = time.time()
print(f"Einsum: {t1 - t0:.4f}s")

print("Same result:", np.all(res1 == res2))

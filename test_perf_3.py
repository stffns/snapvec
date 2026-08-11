import time
import numpy as np

np.random.seed(0)
N = 10000
D = 384
K = 256

X = np.random.randn(N, D).astype(np.float32)

t0 = time.time()
for _ in range(100):
    centers = [X[0]]
    d2 = ((X - centers[0]) ** 2).sum(1)
t1 = time.time()
print(f"Time original kmeans_pp_init: {t1 - t0}")

t0 = time.time()
for _ in range(100):
    centers = [X[0]]
    diff = X - centers[0]
    d2_e = np.einsum('ij,ij->i', diff, diff)
t1 = time.time()
print(f"Time einsum kmeans_pp_init: {t1 - t0}")

print(np.allclose(d2, d2_e))

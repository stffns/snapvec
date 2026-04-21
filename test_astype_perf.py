import numpy as np
import time

def test_redundant_astype():
    np.random.seed(42)
    # create float32 array
    arr = np.random.randn(10000, 384).astype(np.float32)
    rotation = np.random.randn(384, 384).astype(np.float32)

    # 1. matrix multiplication natively float32
    start = time.time()
    for _ in range(100):
        res = arr @ rotation
    print("arr @ rotation: ", time.time() - start)

    # 2. matrix multiplication with redundant astype
    start = time.time()
    for _ in range(100):
        res = (arr @ rotation).astype(np.float32)
    print("(arr @ rotation).astype: ", time.time() - start)

    # 3. simple returning array natively float32
    start = time.time()
    for _ in range(1000):
        res = arr
    print("return arr: ", time.time() - start)

    # 4. simple returning array with redundant astype
    start = time.time()
    for _ in range(1000):
        res = arr.astype(np.float32)
    print("return arr.astype: ", time.time() - start)

test_redundant_astype()

import time

def test_loop(ids, start):
    d = {}
    t0 = time.time()
    for i, id_val in enumerate(ids):
        d[id_val] = start + i
    t1 = time.time()
    return t1 - t0

def test_zip(ids, start):
    d = {}
    t0 = time.time()
    d.update(zip(ids, range(start, start + len(ids))))
    t1 = time.time()
    return t1 - t0

def test_comp(ids):
    t0 = time.time()
    d = {v: i for i, v in enumerate(ids)}
    t1 = time.time()
    return t1 - t0

def test_zip_comp(ids):
    t0 = time.time()
    d = dict(zip(ids, range(len(ids))))
    t1 = time.time()
    return t1 - t0

ids = list(range(1000000))
print("loop:", test_loop(ids, 0))
print("zip update:", test_zip(ids, 0))
print("comp:", test_comp(ids))
print("zip comp:", test_zip_comp(ids))

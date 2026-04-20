# Installation

## From PyPI

```bash
pip install snapvec
```

Pre-compiled wheels are published for CPython 3.10-3.13 on Linux (x86_64,
aarch64), macOS (x86_64 and arm64, macOS 13+), and Windows (AMD64). NumPy
>= 1.24 is the only runtime dependency.

## From source

```bash
git clone https://github.com/stffns/snapvec.git
cd snapvec
pip install -e ".[dev]"
```

On macOS you need OpenMP from Homebrew before building:

```bash
brew install libomp
```

Without `libomp`, the Cython kernels compile in serial mode. The library
still works but the parallel search paths do not.

## Input dtype

`snapvec` expects `np.float32` inputs everywhere. Passing `np.float64`
(the NumPy default for `np.array([...])`) is an error because silently
downcasting would be a surprise on the hot path.

```python
# Models that return float32 directly (most modern embeddings)
vecs = model.encode(texts)  # already float32

# Arrays from np.array([...]) default to float64 -- cast explicitly
vecs = np.array(vecs_list, dtype=np.float32)

# Loading from disk -- respect the original dtype or force float32
vecs = np.load("embeddings.npy").astype(np.float32, copy=False)
```

## Verify install

```python
import snapvec
print(snapvec.__version__)
```

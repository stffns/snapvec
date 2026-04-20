# Concurrency

`snapvec` indexes are **single-writer, multi-reader** within a single
process.

## What's safe

- Multiple threads calling `search()` on the **same** index concurrently.
  Search paths allocate their own scratch buffers and only read shared
  state, so no external lock is required.
- Multiple processes opening **different** index files and querying
  them independently.
- A single writer and any number of readers, as long as you never
  overlap a writer with a reader on the same instance.

## What's not safe

- Two threads calling `add_batch`, `delete`, or `fit` on the same
  index concurrently.  There is no internal lock; the library assumes
  the caller serializes mutations.
- One thread mutating while another searches.  Even when the mutation
  looks atomic at the Python level (for example, appending to a list),
  internal arrays are resized and re-sorted without coordination.

## Recommended pattern

If your application overlaps readers and writers, every public call
must acquire the same lock.  A simple wrapper:

```python
import threading

class SafeIndex:
    def __init__(self, idx):
        self._idx = idx
        self._lock = threading.Lock()

    def add_batch(self, ids, vectors):
        with self._lock:
            self._idx.add_batch(ids, vectors)

    def delete(self, id_):
        with self._lock:
            return self._idx.delete(id_)

    def search(self, query, k=10, **kwargs):
        with self._lock:
            return self._idx.search(query, k=k, **kwargs)
```

If you *never* mutate the index during reads (typical for a
build-then-serve workflow: one `add_batch` at startup, many `search()`
calls forever after), the reader lock can be skipped -- `search()`
only touches immutable shared state (codes, centroids) and
thread-local scratch.  If you need higher read concurrency *and*
occasional writes, use a `threading.RLock` plus a read/write wrapper
(for example, the `readerwriterlock` package) at the application
layer.

## Cross-process access

The on-disk format is designed for cold reload, not shared access:

- `save()` writes to `<path>.tmp` then renames, so a concurrent reader
  calling `load(path)` either sees the old file or the new one, never
  a partial write.
- Nothing prevents two processes from opening the same file and writing
  back.  If you need multi-process writes, put a file lock (for example,
  `fcntl.flock`) around the `save()` call in your application layer.

## Future work

Native single-writer protection via an internal `threading.Lock`, and a
delta-buffer mode for low-latency incremental updates, are tracked on
the roadmap for a future release.

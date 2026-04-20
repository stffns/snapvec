# Filtered search

`SnapIndex` and `IVFPQSnapIndex` accept `filter_ids=<set>` to restrict
results to a subset of ids.  `PQSnapIndex` and `ResidualSnapIndex` do
not yet support this argument; filter the returned list in Python if
you need it.

```python
filter_set = {f"doc-{i:04d}" for i in range(100)}
hits = idx.search(query, k=5, filter_ids=filter_set)
```

## Performance

- **SnapIndex**: the filter is resolved to a sorted row-index slice
  **before** the inner-product matmul, so a sparse filter actively
  reduces scoring work (cost ~ `O(|filter_ids| * dim)` instead of
  `O(N * dim)`).
- **IVFPQSnapIndex** (cluster-aware): probe ranking is restricted to
  clusters that contain at least one filter row, so sparse filters skip
  clusters entirely. Rerank candidates are also drawn from the filtered
  subset, not the unfiltered probe output.

## Edge cases

- Unknown ids in `filter_ids` are silently dropped.
- An entirely-unknown filter returns `[]`.
- A very sparse filter may require a larger `nprobe` on IVF-PQ to surface
  `k` hits.

See [`examples/filter_search.py`](https://github.com/stffns/snapvec/blob/main/examples/filter_search.py)
for a runnable example.

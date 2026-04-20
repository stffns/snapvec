# Save and load

Every index type has `.save(path)` / `.load(path)`.

```python
idx.save("my.snpv")           # SnapIndex
loaded = SnapIndex.load("my.snpv")
```

## Guarantees

- **Atomic writes**: the save target is written to `<path>.tmp` first,
  then renamed. A crash mid-write leaves the old file intact (or absent).
- **CRC32 trailer** (since v0.7): an 8-byte checksum is appended. `load()`
  verifies it; silent corruption raises `ValueError`.
- **Forward-compatible reads**: older files load in newer versions
  wherever the format is documented as backward-compatible; new files
  never claim to be readable by older versions.

## File extensions per index

| Index | Extension | Magic |
|-------|-----------|-------|
| `SnapIndex` | `.snpv` | `SNPV` |
| `PQSnapIndex` | `.snpq` | `SNPQ` |
| `ResidualSnapIndex` | `.snpr` | `SNPR` |
| `IVFPQSnapIndex` | `.snpi` | `SNPI` |

Paths can use any extension -- the magic header determines the format.
The canonical extensions are a convention for your tooling.

## IDs

`ids` can be any hashable Python value. They are serialized as strings
in the file and round-tripped through `load`. If you pass integers,
they come back as integers; strings come back as strings.

See [`examples/save_load.py`](https://github.com/stffns/snapvec/blob/main/examples/save_load.py)
for a runnable example.

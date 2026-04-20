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

`ids` can be any hashable Python value.  They are serialized as strings
in the file; on `load()`, `snapvec` tries to decode each string as
`int`, then `float`, and falls back to the raw string.  That means:

- `42` -> `42` (int round-trip).
- `3.14` -> `3.14` (float round-trip).
- `"abc"` -> `"abc"` (string round-trip).
- `"123"` -> `123` (the string form of a number is loaded back as `int`,
  not `str`).  If you need to preserve a numeric-looking string verbatim,
  prefix it (for example, `"id-123"`) before ingesting.

See [`examples/save_load.py`](https://github.com/stffns/snapvec/blob/main/examples/save_load.py)
for a runnable example.

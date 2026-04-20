# snapvec -- Claude-specific instructions

## Commits
- NEVER add `Co-Authored-By` lines to commit messages.
- Use conventional prefixes: `feat`, `fix`, `chore`, `docs`, `bench`, `refactor`, `test`.
- Keep subject under 72 chars. Present-tense imperative.

## Code style
- Prefer ASCII in new or edited text (source, docstrings, docs, commits).
  No em dashes, no smart quotes, no ellipsis character. Use `-`, `--`,
  `"`, `'`, `...`.  Existing math notation in docstrings (`sqrt`, arrows,
  inner-product brackets) may remain unless you are already rewriting
  that block.
- English everywhere in code, docstrings, commits, and README.

## Project layout
- `snapvec/` = library (public API in `__init__.py`).
- `tests/` = pytest suite (151 tests as of v0.9.0).
- `experiments/` = rough benchmarks and profiling scripts -- do not treat as first-class code.
- `bench/` (future) = reproducible benchmark suite that runs in CI.

## Running checks locally
```bash
ruff check snapvec/ tests/
pytest -q
mypy --strict snapvec/   # 17 errors as of 2026-04-20, warning-only in CI
```

## File format invariants
- Any change to the on-disk format (`.snpv`, `.snpq`, `.snpr`, `.snpi`) must bump
  the format version in `snapvec/_file_format.py` and add a round-trip test.
- Do not remove or reorder existing serialized fields without a migration path.

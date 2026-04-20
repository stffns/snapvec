# Contributing to snapvec

Thanks for your interest. snapvec is a small, focused library. Contributions
are welcome; please read this before opening a PR so we can keep the feedback
loop short.

## Getting started

```bash
git clone https://github.com/stffns/snapvec.git
cd snapvec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

On macOS you also need OpenMP:

```bash
brew install libomp
```

## Running the test suite

```bash
pytest -q
```

Coverage:

```bash
pytest --cov=snapvec --cov-report=term-missing
```

## Linting and types

```bash
ruff check snapvec/ tests/
mypy --strict snapvec/
```

`ruff format` is intentionally not used: it rewrites the hand-aligned
codebook constants in `snapvec/_codebooks.py` into a less readable layout.
`pre-commit` runs `ruff check` on every commit.

## Benchmarks

Reproducible benchmarks live under `experiments/`. If you add or modify one,
include the hardware (`uname -a`, CPU model), NumPy version, and the commit
hash in the output so results are comparable across runs.

## Pull requests

- Keep the change focused. One PR, one idea.
- Add or update tests for every behavior change.
- If the change affects performance, include a before/after measurement.
- If the change affects the on-disk file format, bump the format version in
  `snapvec/_file_format.py` and add a round-trip test.
- Update `CHANGELOG.md` under the `[Unreleased]` section.

## Commit style

- Use present tense imperative ("add X", "fix Y", not "added" / "fixes").
- Keep the subject line under 72 characters.
- Reference the issue number in the body if applicable.

## Release process

Releases are cut by the maintainer:
1. Bump version in `pyproject.toml`.
2. Move `[Unreleased]` entries to a new version heading in `CHANGELOG.md`.
3. Tag `vX.Y.Z` and push.
4. The `Release` workflow builds wheels and publishes to PyPI via
   trusted publishing.

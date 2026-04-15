# `docs/blog/`

Long-form drafts that go alongside snapvec releases.  Kept in the
repo so the source-of-truth versions live next to the code and
benchmarks they cite — copies live wherever they get published
(dev.to, personal blog, HN, etc.) but the canonical text is here.

## Posts

| # | Title | Status | Tied to release |
|---:|---|---|---|
| 01 | [I tried 6 NumPy-level optimizations on a vector index. 4 lost. Here's the math.](./01-numpy-perf-ceiling.md) | draft, ready to publish | v0.5.0 |
| 02 | [Then we added Numba. Here's what changed.](./02-numba-followup.md) | outline, fill after Numba lands | v0.7.0 |

## Style notes

- Lead with the table.  HN scrolls past anything that looks like
  setup before the punchline.
- Negative results are the asset.  Positive results in NumPy land
  are common; "tried X, lost, here's why" is rare and useful.
- Keep numbers reproducible from the `experiments/` scripts.  Cite
  exact tag (e.g., `v0.5.0`).
- Prose voice: technical-honest, no marketing.  The audience is
  people who'd run our benchmark on their laptop to verify.

## Distribution checklist (when publishing post 01)

- [ ] dev.to (canonical platform for Python perf content)
- [ ] HN with title "I tried 6 NumPy-level optimizations on a
      vector index. 4 of them lost."
- [ ] Update snapvec README with a "Reading" section linking the post
- [ ] Cross-post on personal blog if applicable
- [ ] Tweet/Mastodon with the screenshot of the result table

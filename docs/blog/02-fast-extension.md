# Then we shipped a `[fast]` extension.  Here's what changed — and which compiler we picked, and why.

*Outline — author: Jay (stffns) · TBD after the fast extension lands*

> **Status:** outline only.  Numbers and the implementation-language
> decision will land once the extension is shipped.  This file is the
> structural draft so the post is a quick fill-in-the-blanks job.
>
> The decision lives at the bottom (Numba vs Rust); the post body
> describes whichever path we picked.

---

## Hook

Quote the closing line of the v0.5 post:
> *"The only thing that removes the dispatch is moving the per-
> subspace loop inside C."*

Then immediately:
> So I did.  And here's what actually changed when the dispatch
> floor disappeared — plus why I picked [`Numba` | `Rust + PyO3`]
> over the obvious alternative.

## The setup (1 paragraph)

Recap from post #1:

- snapvec's IVF-PQ search hot path was dispatch-bound at M=192 in
  pure NumPy.
- 6 NumPy-level optimizations attempted, 4 lost for the same
  structural reason.
- Final v0.5 ms/q at 0.91 recall ≈ 1.5 ms.  Floor: ~5 µs / candidate.
- v0.6/v0.7 promise: an opt-in `snapvec[fast]` extra with a
  compiled inner loop.  This is the post about that extra.

## The kernel (TODO once shipped)

Show the actual code — likely ~30-40 lines either way.  Annotate:

- The per-subspace gather + sum becomes a single tight loop in C-
  level memory.
- The algorithm is identical; only the implementation language and
  the toolchain change.
- Whichever language we picked, the loop ends up being SIMD
  vectorised — explicitly with intrinsics if Rust, by the LLVM
  backend if Numba.

Compare side-by-side: NumPy version of `_score_one` vs the compiled
version.

## Numbers (TODO)

| nprobe | recall | ms (NumPy v0.5) | ms (`[fast]`) | speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.771 | 0.59 | _TBD_ | _TBD_ |
| 32 | 0.891 | 1.55 | _TBD_ | _TBD_ |
| 64 | 0.914 | 2.70 | _TBD_ | _TBD_ |
| 128 | 0.924 | 4.96 | _TBD_ | _TBD_ |

Plus N=1M extrapolation, since that was the original sprint target.

If we hit < 1 ms at the 0.91 recall point: lead with that headline.

## The Numba vs Rust decision

This is the part the post hangs its honesty on.  The decision was
*not* obvious; both paths have real trade-offs.  Walk readers
through the same matrix we used:

| Dimension | Numba | Rust + PyO3 + maturin |
|---|---|---|
| Time to first speedup | hours | days (PyO3 learning curve) |
| First-call latency | JIT warmup ~hundreds of ms | zero (AOT) |
| Wheel distribution | requires `numba` runtime + LLVM | precompiled wheels per platform |
| SIMD control | implicit (LLVM picks) | explicit (`std::arch` / `portable_simd`) |
| CI matrix | works on the dev's box, sometimes silently broken on Linux ARM / Alpine / niche Pythons | proper `cibuildwheel` matrix, painful to set up but predictable once it works |
| Reference projects | `umap-learn`, `librosa` perf paths | Polars, Ruff, HF `tokenizers`, `pydantic-core` |
| Iteration speed for further perf work | edit the Python file, re-run | edit Rust, recompile, re-bind |

The pragmatic recommendation we ended up at:

> **Numba first** to validate the speedup is real and worth the
> packaging investment — it's a few hours and we know in
> milliseconds whether the dispatch-floor argument from post #1
> was actually right.
>
> **Rust via PyO3 later** once we know snapvec is around long
> enough to justify the wheel-matrix investment.  Polars / Ruff /
> tokenizers built their distribution stories around exactly this
> pattern; if snapvec grows past hobby scale, we owe its users the
> same guarantees (no JIT warmup, no LLVM on the user's box, real
> SIMD intrinsics, predictable CI).

If we shipped Numba: post explains why "validate before
productize", with the speedup numbers as the validation.  If we
shipped Rust: post explains why we skipped the Numba prototype
(probably because the project felt mature enough to invest in the
packaging straight away).

## What did NOT change

- The recall ceiling.  This is a PQ-rate property, independent of
  implementation.  v0.6's `pq_rerank` is what unblocks recall ≥
  0.94; the fast extension is what unblocks latency.  Two
  orthogonal axes; both addressed in their own release.
- The default install.  `pip install snapvec` is still single-numpy,
  zero deps.  `pip install snapvec[fast]` adds the chosen
  accelerator.

## Open questions for the next sprint

- Direct SIMD intrinsics for PQ4 / fast-scan (FAISS's secret sauce):
  AVX2 `pshufb` or NEON `tbl` for 16 lookups per instruction.  Real
  if we're already in Rust; harder if we're in Numba.
- Mixed precision (int8 codes + int16 accumulators).  Was a loss in
  NumPy because of dispatch overhead; almost certainly a win in any
  compiled inner loop.
- An IVF coarse partition that uses HNSW instead of flat k-means,
  since that's the next bottleneck once the per-subspace loop is
  free.

## Closing

> I started this sprint thinking the negative findings would feel
> like dead ends.  They didn't — they made the next decision
> obvious.  When the only path forward is "drop into a compiled
> inner loop", the choice between Numba and Rust stops being a
> question of *which is fancier* and becomes a question of *which
> matches your project's lifecycle*.  Both are correct answers;
> they're just answers to different questions.

Link to release notes, install instructions, FIQA bench
reproduction script.

---

## Distribution checklist (when posting)

- [ ] dev.to (Python perf / packaging audience)
- [ ] HN: title depends on which path we picked.  If Numba:
      "I added Numba to my pure-NumPy vector index. The 4 negative
      results from last sprint are now positive."  If Rust:
      "I rewrote my vector index hot path in Rust.  Here's why
      Numba was the wrong tool."
- [ ] Cross-post on personal blog if applicable
- [ ] Update snapvec README with link to both posts
- [ ] If Rust path: also r/rust and possibly the PyO3 community

# Then we added Numba.  Here's what changed.

*Outline — author: Jay (stffns) · TBD after v0.7 ships*

> **Status:** outline only.  Numbers will land once the `numba`
> kernel is shipped in v0.7.  This file is the structural draft so
> the post is a quick fill-in-the-blanks job, not a from-scratch
> write-up after the work.

---

## Hook

Quote the closing line of the v0.6 post:
> *"The only thing that removes the dispatch is moving the per-
> subspace loop inside C."*

Then immediately:
> So I did.  And here's what actually changed when the dispatch
> floor disappeared.

## The setup (1 paragraph)

Recap from post #1:

- snapvec's IVF-PQ search hot path was dispatch-bound at M=192 in
  pure NumPy.
- 6 NumPy-level optimizations attempted, 4 lost for the same
  structural reason.
- Final v0.5 ms/q at 0.91 recall ≈ 1.5 ms.  Floor: ~5 µs / candidate.
- v0.6 promise: an opt-in `snapvec[fast]` extra with a Numba kernel
  to remove the dispatch.

## The kernel (TODO once shipped)

Show the actual `@numba.njit` code — likely ~30 lines.  Annotate:

- The per-subspace gather + sum becomes a single tight loop.
- Numba's typing inference + LLVM backend lets it auto-vectorize.
- No quantisation needed for the win — float32 is fine once the
  loop runs in C.

Compare side-by-side: NumPy version vs Numba version of `_score_one`.
Highlight that the algorithm is identical; only the implementation
language changes.

## Numbers (TODO)

A table comparing the v0.5 NumPy baseline against v0.7 with
`snapvec[fast]` at the same FIQA recall sweep:

| nprobe | recall | ms (NumPy) | ms (Numba) | speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.771 | 0.59 | _TBD_ | _TBD_ |
| 32 | 0.891 | 1.55 | _TBD_ | _TBD_ |
| 64 | 0.914 | 2.70 | _TBD_ | _TBD_ |
| 128 | 0.924 | 4.96 | _TBD_ | _TBD_ |

Plus the N=1M extrapolation table, since that was the original
sprint target.

If we hit < 1 ms at the 0.91 recall point: lead with that headline.
If we hit ~1-2 ms: lead with "we hit the BLAS-equivalent regime
without writing C ourselves" — also a good story.

## What I learned about Numba (educational meat)

Things that are NOT obvious until you do it:

1. **First-call JIT cost** (~hundreds of ms).  How to handle in a
   library: warm-up at import? Lazy on first search?  AOT compile?
2. **Type signatures matter.**  Numba decides what to compile based
   on the first call's argument types.  Lock them down explicitly to
   avoid recompilation on every dtype variation.
3. **Cache to disk** with `cache=True` — second-run startup matters.
4. **Parallel mode** (`parallel=True` + `prange`).  When does it
   help?  When does it conflict with our outer thread pool?
5. **Memory layout still matters.**  The column-major `(M, n)` layout
   we chose in v0.5 turns out to be exactly what the Numba loop wants
   too — vindication.

## What did NOT change

- The recall ceiling.  This is a PQ-rate property, independent of
  implementation.  v0.6's `pq_rerank` is what unblocks recall ≥ 0.94;
  v0.7's Numba is what unblocks latency.  They are orthogonal wins.
- The default install.  `pip install snapvec` is still single-numpy,
  zero deps.  `pip install snapvec[fast]` adds numba.

## What I'd try next (open questions for v0.8+)

- Cython instead of Numba: the JIT cost goes away but the build
  story gets more involved (wheels per platform, `cibuildwheel`).
- Direct SIMD intrinsics (AVX2 / NEON) via Cython's `cdef` — what
  PQ4 / fast-scan in FAISS does.  Probably another 2-3× over Numba,
  but the code becomes platform-conditional.
- Mixed precision: int8 codes + int16 accumulators, *now* that the
  inner loop is in C and the dispatch overhead is gone.  Was a loss
  in NumPy; almost certainly a win in Numba.

## Closing

> Negative findings in a blog post don't cost anything if you've
> already paid for them with the experiments.  Documenting that
> NumPy hit a floor at 1.5 ms made it obvious where the next dollar
> of effort should go.  And the moment we crossed the floor, the
> result was concrete enough to put in the README without an
> asterisk.

Link to v0.7 release, `snapvec[fast]` install instructions, FIQA
bench reproduction script.

---

## Distribution checklist (when posting)

- [ ] dev.to (most ANN library users land here)
- [ ] HN: title "I added Numba to my pure-NumPy vector index. The 4
      negative results from last sprint are now positive."
- [ ] Twitter/X with the speedup chart
- [ ] Cross-post on personal blog if applicable
- [ ] Update snapvec README with link to both posts

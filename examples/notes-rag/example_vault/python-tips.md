---
tags: [python, tips]
---

# Python performance notes

## When NumPy beats pure Python

Python-level loops pay a dispatch cost per iteration. If you can express a
computation as a single array op — elementwise arithmetic, reductions,
reshape + broadcast — NumPy runs it in C with SIMD.  The rule of thumb I
use: if your inner loop is doing a fixed arithmetic pattern over N items,
there is almost always a vectorised form.

The tricky part is resisting the urge to "just add another for-loop" when
the vectorised form is slightly harder to read.  A tensor view via
`.reshape(..., k, 2, h)` can collapse nested loops into one expression.

## Avoid float64 when you have float32 data

`np.array([1.0, 2.0])` gives you float64 by default.  If you then pass it
to a library that operates in float32 internally, the library will copy
the whole thing to cast.  Cheap fix: `np.asarray(..., dtype=np.float32)`
at the boundary.

## Tags

This file is tagged `#python` and `#tips`, which makes it searchable by
`notes-rag ask "..." --tag python`.

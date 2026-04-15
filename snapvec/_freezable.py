"""Mutation-guard mixin + thread-safe search contract.

``FreezableIndex`` is tiny by design: a boolean flag, a pair of
freeze/unfreeze methods, and a ``_check_not_frozen`` helper the
mutator methods call before doing any work.  The thread-safety
contract is: **after** ``idx.freeze()`` is called, the index's
mutable state does not change, so reads from multiple threads are
free of races without any locks on the hot path.

This is deliberately simpler than a full reader-writer lock —
production deployments of snapvec are "build offline, query online"
with very few live mutations, and paying a lock on every ``search()``
call to serve the rare mutation case is the wrong trade.  Users that
do need live mutation during serving can serialise it externally or
hold a short mutex between ``unfreeze`` / mutate / ``freeze`` cycles.

Each index class mixes this in and calls ``self._check_not_frozen()``
at the top of each mutating method (``add_batch``, ``delete``, ``fit``,
``close``, …).
"""
from __future__ import annotations


class FreezableIndex:
    """Mixin that gives an index opt-in immutability + thread-safe reads."""

    _frozen: bool = False

    @property
    def frozen(self) -> bool:
        """True when the index is in its read-only, thread-safe state."""
        return self._frozen

    def freeze(self) -> None:
        """Mark the index immutable.

        After ``freeze()``, calls to ``add`` / ``add_batch`` / ``delete``
        / ``fit`` / ``close`` raise ``RuntimeError``.  Concurrent
        ``search()`` from multiple threads is then safe **by contract**
        — no internal state changes underneath the readers, so no lock
        is taken on the query hot path.

        Typical usage::

            idx = IVFPQSnapIndex(...)
            idx.fit(sample); idx.add_batch(ids, vectors)
            idx.freeze()                         # ← done mutating
            # serve queries from any number of threads
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable mutations.  Rarely needed outside test harnesses.

        Note: while the index is unfrozen, concurrent mutation +
        search is not thread-safe.  Callers are responsible for
        serialising access during the mutation window.
        """
        self._frozen = False

    def _check_not_frozen(self, op: str = "this operation") -> None:
        if self._frozen:
            raise RuntimeError(
                f"{op} is not allowed on a frozen index.  Call "
                f"idx.unfreeze() first if you need to mutate, then "
                f"idx.freeze() again when you're done."
            )


__all__ = ["FreezableIndex"]

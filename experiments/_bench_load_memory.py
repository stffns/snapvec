"""Measure peak RSS during PQSnapIndex.load() and IVFPQSnapIndex.load().

Quick sanity check for the Fix B load() optimization -- verifies the
readinto path does not double the on-disk size during load.

Uses a subprocess per load so ru_maxrss is measured against a clean
baseline (getrusage tracks high-water mark, so in-process measurements
are dominated by the fit phase).
"""
from __future__ import annotations

import argparse
import os
import pickle
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if rss > 1_000_000 else rss / 1024


def _child_load(kind: str, path: str) -> None:
    from snapvec import IVFPQSnapIndex, PQSnapIndex
    if kind == "pq":
        loaded = PQSnapIndex.load(path)
    elif kind == "ivfpq":
        loaded = IVFPQSnapIndex.load(path)
    else:
        raise ValueError(kind)
    peak = _rss_mb()
    print(f"CHILD_PEAK={peak:.2f} codes_mb={loaded._codes.nbytes/(1024*1024):.2f}")


def _make_pq(path: Path, n: int, dim: int, M: int) -> float:
    from snapvec import PQSnapIndex
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    idx = PQSnapIndex(dim=dim, M=M, seed=0, normalized=True)
    idx.fit(vecs[: min(20_000, n)])
    idx.add_batch([str(i) for i in range(n)], vecs)
    idx.save(path)
    return path.stat().st_size / (1024 * 1024)


def _make_ivfpq(
    path: Path, n: int, dim: int, M: int, nlist: int,
) -> float:
    from snapvec import IVFPQSnapIndex
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    idx = IVFPQSnapIndex(dim=dim, M=M, nlist=nlist, seed=0, normalized=True)
    idx.fit(vecs[: min(20_000, n)])
    idx.add_batch([str(i) for i in range(n)], vecs)
    idx.save(path)
    return path.stat().st_size / (1024 * 1024)


def _run_child(kind: str, path: Path) -> tuple[float, float]:
    out = subprocess.check_output(
        [sys.executable, __file__, "--child", kind, str(path)],
        text=True,
    )
    peak = 0.0
    codes = 0.0
    for line in out.splitlines():
        if line.startswith("CHILD_PEAK="):
            parts = dict(kv.split("=") for kv in line.split())
            peak = float(parts["CHILD_PEAK"])
            codes = float(parts["codes_mb"])
    return peak, codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=2, default=None)
    args = parser.parse_args()
    if args.child is not None:
        _child_load(args.child[0], args.child[1])
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for n in (200_000, 500_000):
            pq_path = tmpdir / f"pq_{n}.snpq"
            pq_file = _make_pq(pq_path, n=n, dim=384, M=96)
            peak, codes = _run_child("pq", pq_path)
            print(
                f"PQ     n={n//1000:>3}k  M=96   file={pq_file:6.1f} MB  "
                f"codes={codes:6.1f} MB  peak RSS={peak:6.1f} MB  "
                f"overhead={peak-codes:+5.1f} MB"
            )

            ivf_path = tmpdir / f"ivfpq_{n}.snpi"
            ivf_file = _make_ivfpq(ivf_path, n=n, dim=384, M=192, nlist=64)
            peak, codes = _run_child("ivfpq", ivf_path)
            print(
                f"IVFPQ  n={n//1000:>3}k  M=192  file={ivf_file:6.1f} MB  "
                f"codes={codes:6.1f} MB  peak RSS={peak:6.1f} MB  "
                f"overhead={peak-codes:+5.1f} MB"
            )


if __name__ == "__main__":
    main()

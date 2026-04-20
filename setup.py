"""Build script for Cython extensions with OpenMP support."""
import platform
import subprocess
import sys

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize


def _openmp_flags():
    """Detect OpenMP flags for the current platform."""
    if sys.platform == "darwin":
        # macOS: clang needs libomp from Homebrew
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return (
                ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                ["-lomp", f"-L{prefix}/lib"],
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No Homebrew libomp -- fall back to serial
            return ([], [])
    elif sys.platform == "win32":
        return (["/openmp"], [])
    else:
        # Linux / other POSIX
        return (["-fopenmp"], ["-fopenmp"])


def _arch_flags():
    """Host-optimized CPU target flags per platform.

    GCC/clang on x86 accept ``-march=native``.  On ARM (Apple Silicon,
    Graviton), clang expects ``-mcpu=native`` instead; passing
    ``-march=native`` triggers ``unknown target CPU 'apple-m1'`` on the
    Xcode-bundled clang that GitHub macos-14 runners ship.  MSVC doesn't
    have an equivalent knob.
    """
    if sys.platform == "win32":
        return []
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return ["-mcpu=native"]
    return ["-march=native"]


compile_flags, link_flags = _openmp_flags()
arch_flags = _arch_flags()

extensions = [
    Extension(
        "snapvec._fast",
        sources=["snapvec/_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"] + arch_flags + compile_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)

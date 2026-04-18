"""Build script for Cython extensions with OpenMP support."""
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


compile_flags, link_flags = _openmp_flags()

extensions = [
    Extension(
        "snapvec._fast",
        sources=["snapvec/_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"] + compile_flags,
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

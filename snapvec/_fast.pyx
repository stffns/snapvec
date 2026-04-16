# cython: boundscheck=False, wraparound=False, cdivision=True
"""Compiled kernels for snapvec hot paths (Cython + OpenMP)."""

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef Py_ssize_t _N_PARALLEL_THRESHOLD = 2000


def adc_colmajor(
    const float[:, :] lut,
    const unsigned char[:, :] codes,
    float[:] scores,
    bint parallel=False,
):
    cdef Py_ssize_t M = codes.shape[0]
    cdef Py_ssize_t n = codes.shape[1]
    cdef Py_ssize_t i, j
    cdef float acc
    cdef float* out = &scores[0]

    if parallel and n >= _N_PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True, schedule="static"):
            acc = out[i]
            for j in range(M):
                acc = acc + lut[j, codes[j, i]]
            out[i] = acc
    else:
        with nogil:
            for i in range(n):
                acc = out[i]
                for j in range(M):
                    acc = acc + lut[j, codes[j, i]]
                out[i] = acc


def fused_gather_adc(
    const unsigned char[:, :] all_codes,
    const long[:] row_idx,
    const float[:] coarse_offsets,
    const float[:, :] lut,
    float[:] scores,
    bint parallel=True,
):
    cdef Py_ssize_t M = all_codes.shape[0]
    cdef Py_ssize_t n = row_idx.shape[0]
    cdef Py_ssize_t i, j
    cdef long r
    cdef float acc
    cdef float* out = &scores[0]

    if parallel and n >= _N_PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True, schedule="static"):
            r = row_idx[i]
            acc = coarse_offsets[i]
            for j in range(M):
                acc = acc + lut[j, all_codes[j, r]]
            out[i] = acc
    else:
        with nogil:
            for i in range(n):
                r = row_idx[i]
                acc = coarse_offsets[i]
                for j in range(M):
                    acc = acc + lut[j, all_codes[j, r]]
                out[i] = acc

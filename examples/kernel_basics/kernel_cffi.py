# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import os
import time
import numpy as np
import cffi

ffi_interface = cffi.FFI()

src = r"""
#include <math.h>

void mymul(int n,
    double* x1, double* x2,
    double* y){

   for (int ii=0; ii<n; ii++){
      y[ii] = x1[ii] * x2[ii];
   }
}
"""

ffi_interface.cdef(
    """
   void mymul(int n,
       double* x1, double* x2,
       double* y);"""
)


ffi_interface.set_source(
    "_example",
    src,
    extra_compile_args=["-O3"],
    extra_link_args=["-O3"],
)

ffi_interface.compile(verbose=True)

# potentially stopping interpreter

from _example import ffi, lib

x1 = np.array([1, 2, 3, 4], dtype=np.float64)
x2 = np.array([7, 8, 9, 10], dtype=np.float64)
y = np.zeros((len(x1),), dtype=np.float64)
x1_cffi = ffi.cast("double *", ffi.from_buffer(x1))
x2_cffi = ffi.cast("double *", ffi.from_buffer(x2))
y_cffi = ffi.cast("double *", ffi.from_buffer(y))

lib.mymul(len(x1), x1_cffi, x2_cffi, y_cffi)

assert np.allclose(y, x1 * x2)

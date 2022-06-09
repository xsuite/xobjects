# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

source_str = r"""
__kernel
void mymul(int n,
    __global const double* x1,
    __global const double* x2,
    __global double* y)
{
    int tid = get_global_id(0);
    y[tid] = x1[tid] * x2[tid];

}"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, source_str).build()
mymul_kernel = prg.mymul

x1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
x2 = np.array([7, 8, 9, 10, 12], dtype=np.float64)

x1_dev = cl_array.to_device(queue, x1)
x2_dev = cl_array.to_device(queue, x2)
y_dev = cl_array.zeros(queue, len(x1_dev), dtype=np.float64)

mymul_kernel(
    queue,
    (len(x1),),
    None,
    np.int32(len(x1)),
    x1_dev.data,
    x2_dev.data,
    y_dev.data,
)

y = y_dev.get()

assert np.allclose(y, x1 * x2)

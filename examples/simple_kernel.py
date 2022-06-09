# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo

import logging

logging.basicConfig(level=logging.DEBUG)

src_code = r"""
double my_mul(const int n, const double* x1,
            const double* x2) {
    int tid;
    double y =0;
    for (tid=0; tid<n; tid++){
        y+= x1[tid] * x2[tid];
        }
    return y;
    }
"""
kernel_descriptions = {
    "my_mul": {
        "args": (
            (
                ("scalar", np.int32),
                "n",
            ),
            (
                ("array", np.float64),
                "x1",
            ),
            (
                ("array", np.float64),
                "x2",
            ),
        ),
        "return": ("scalar", np.float64),
        "num_threads_from_arg": "n",
    },
}

# create context
context = xo.ContextCpu()

# Import kernel in context
context.add_kernels(src_code=src_code, kernel_descriptions=kernel_descriptions)

# With a1 and a2 being arrays on the context, the kernel

a1 = np.arange(10.0)
a2 = np.arange(10.0)

# can be called as follows:
y = context.kernels.my_mul(n=len(a1), x1=a1, x2=a2)

print(y)

import numpy as np

import xobjects as xo
from xobjects.context import available


def test_kernel_cpu():
    ctx = xo.ContextCpu()
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

    ctx.add_kernels(src_code=src_code, kernel_descriptions=kernel_descriptions)
    a1 = np.arange(10.0)
    a2 = np.arange(10.0)
    y = ctx.kernels.my_mul(n=len(a1), x1=a1, x2=a2)

    assert y == 285.0

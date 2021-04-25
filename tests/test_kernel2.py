import numpy as np

import xobjects as xo
from xobjects.context import available


def test_kernel_cpu():
    ctx = xo.ContextCpu()
    source = r"""
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
    kernels = {
        "my_mul": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, pointer=True, name="x1"),
                xo.Arg(xo.Float64, pointer=True, name="x2"),
            ],
            ret=xo.Arg(xo.Float64),
        )
    }

    ctx.add_kernels_v2(sources=[source], kernels=kernels)
    a1 = np.arange(10.0)  # TODO: check context
    a2 = np.arange(10.0)
    y = ctx.kernels.my_mul(n=len(a1), x1=a1, x2=a2)

    assert y == 285.0

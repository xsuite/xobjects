# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


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
        "my_mul": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, pointer=True, name="x1"),
                xo.Arg(xo.Float64, pointer=True, name="x2"),
            ],
            ret=xo.Arg(xo.Float64),
        )
    }

    ctx.add_kernels(sources=[src_code], kernels=kernel_descriptions) 
    a1 = np.arange(10.0)
    a2 = np.arange(10.0)
    y = ctx.kernels.my_mul(n=len(a1), x1=a1, x2=a2)

    assert y == 285.0


@for_all_test_contexts
def test_kernels(test_context):
    src_code = """
    /*gpufun*/
    void myfun(double x, double y,
        double* z){
        z[0] = x * y;
        }

    /*gpukern*/
    void my_mul(const int n,
        /*gpuglmem*/ const double* x1,
        /*gpuglmem*/ const double* x2,
        /*gpuglmem*/       double* y) {
        int tid = 0 //vectorize_over tid n
        double z;
        myfun(x1[tid], x2[tid], &z);
        y[tid] = z;
        //end_vectorize
        }
    """

    kernel_descriptions = {
        "my_mul": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, pointer=True, const=True, name="x1"),
                xo.Arg(xo.Float64, pointer=True, const=True, name="x2"),
                xo.Arg(xo.Float64, pointer=True, const=False, name="y"),
            ],
            n_threads="n",
        ),
    }

    # Import kernel in context
    test_context.add_kernels(
        sources=[src_code],
        kernels=kernel_descriptions,
        # save_src_as=f'_test_{name}.c')
        save_source_as=None,
        compile=True
    )

    x1_host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2_host = np.array([7.0, 8.0, 9.0], dtype=np.float64)

    x1_dev = test_context.nparray_to_context_array(x1_host)
    x2_dev = test_context.nparray_to_context_array(x2_host)
    y_dev = test_context.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

    test_context.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

    y_host = test_context.nparray_from_context_array(y_dev)

    assert np.allclose(y_host, x1_host * x2_host)

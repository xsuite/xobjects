# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2025.                   #
# ########################################### #

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_common_atomicadd(test_context):
    src = r"""
    #include <xobjects/headers/common.h>

    GPUKERN
    double test_atomic_add()
    {
        int iterations = 1000;
        double sum = 0;
        VECTORIZE_OVER(i, iterations);
            // If on CPU do some work to avoid the loop being optimized out
            #if defined(XO_CONTEXT_CPU_OPENMP)
                usleep(10);
            #endif
            atomicAdd(&sum, 1.0);
        END_VECTORIZE;
        return sum;
    }
    """

    n_threads = 1
    if type(test_context).__name__ in {"ContextCupy", "ContextPyopencl"}:
        n_threads = 1000
    elif (
        test_context.omp_num_threads == "auto"
        or test_context.omp_num_threads > 1
    ):
        n_threads = 8

    test_context.add_kernels(
        sources=[src],
        kernels={
            "test_atomic_add": xo.Kernel(
                args=[],
                n_threads=n_threads,
                ret=xo.Arg(xo.Float64),
            )
        },
    )

    expected = 1000
    result = test_context.kernels.test_atomic_add()

    assert result == expected

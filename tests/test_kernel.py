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


def test_kernels():
    for (CTX, kwargs, name) in zip(
            (xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy),
            ({'omp_num_threads': 2}, {}, {}),
            ('cpu', 'pyopencl', 'cupy')):

        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX(**kwargs)

        src_code='''
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
        '''

        kernel_descriptions = {'my_mul':{
            'args':(
                (('scalar', np.int32),   'n',),
                (('array',  np.float64), 'x1',),
                (('array',  np.float64), 'x2',),
                (('array',  np.float64), 'y',),
                ),
            'num_threads_from_arg': 'n'
            },}

        # Import kernel in context
        ctx.add_kernels(src_code=src_code,
                kernel_descriptions=kernel_descriptions,
                #save_src_as=f'_test_{name}.c')
                save_src_as=None)

        x1_host = np.array([1.,2.,3.], dtype=np.float64)
        x2_host = np.array([7.,8.,9.], dtype=np.float64)

        x1_dev = ctx.nparray_to_context_array(x1_host)
        x2_dev = ctx.nparray_to_context_array(x2_host)
        y_dev = ctx.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

        ctx.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

        y_host = ctx.nparray_from_context_array(y_dev)

        assert np.allclose(y_host, x1_host*x2_host)

# With a1 and a2 being arrays on the context, the kernel
# can be called as follows:
#conddtext.kernels.my_mul(n=len(a1), x1=a1, x2=a2)

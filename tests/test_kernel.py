# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import cffi
import sysconfig

import numpy as np

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts, requires_context


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
        compile=True,
    )

    x1_host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2_host = np.array([7.0, 8.0, 9.0], dtype=np.float64)

    x1_dev = test_context.nparray_to_context_array(x1_host)
    x2_dev = test_context.nparray_to_context_array(x2_host)
    y_dev = test_context.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

    test_context.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

    y_host = test_context.nparray_from_context_array(y_dev)

    assert np.allclose(y_host, x1_host * x2_host)


@for_all_test_contexts
def test_kernels_manual_add(test_context):
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
    kernels = test_context.build_kernels(
        sources=[src_code],
        kernel_descriptions=kernel_descriptions,
        save_source_as=None,
        compile=True,
    )
    test_context.kernels.update(kernels)

    x1_host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2_host = np.array([7.0, 8.0, 9.0], dtype=np.float64)

    x1_dev = test_context.nparray_to_context_array(x1_host)
    x2_dev = test_context.nparray_to_context_array(x2_host)
    y_dev = test_context.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

    test_context.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

    y_host = test_context.nparray_from_context_array(y_dev)

    assert np.allclose(y_host, x1_host * x2_host)


@requires_context("ContextCpu")
def test_kernels_save_files(tmp_path):
    test_context = xo.ContextCpu()
    my_folder = tmp_path / "my_folder"

    src_code = """
        /*gpufun*/
        double myfun(double x, double y){
            return x * y;
        }
        """

    kernel_descriptions = {
        "myfun": xo.Kernel(
            args=[
                xo.Arg(xo.Float64, name="x"),
                xo.Arg(xo.Float64, name="y"),
            ],
            ret=xo.Arg(xo.Float64),
        ),
    }

    kernels = test_context.build_kernels(
        sources=[src_code],
        kernel_descriptions=kernel_descriptions,
        save_source_as="my_module.c",
        compile=True,
        containing_dir=str(my_folder),
        module_name="my_module",
    )
    test_context.kernels.update(kernels)

    assert test_context.kernels.myfun(x=3, y=4) == 12

    assert my_folder.exists()
    so_file = my_folder / (
        "my_module" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    assert so_file.exists()
    assert (my_folder / "my_module.c").exists()


@requires_context("ContextCpu")
def test_kernels_save_load_with_classes(tmp_path, mocker):
    """Test the use case of xtrack.

    We build a class with a kernel, verify that the kernel works, and then we
    save the kernel to a file. We then reload the kernel from the file and
    verify that it still works on a fresh context.
    """
    test_context = xo.ContextCpu()
    my_folder = tmp_path / "my_folder"

    class TestClass(xo.HybridClass):
        _xofields = {
            "x": xo.Float64,
            "y": xo.Float64,
        }
        _extra_c_sources = [
            """
            /*gpufun*/ double myfun(TestClassData tc){
                double x = TestClassData_get_x(tc);
                double y = TestClassData_get_y(tc);
                return x * y;
            }
        """
        ]
        _kernels = {
            "myfun": xo.Kernel(
                args=[
                    xo.Arg(xo.ThisClass, name="tc"),
                ],
                ret=xo.Arg(xo.Float64),
            ),
        }

    kernels = test_context.build_kernels(
        kernel_descriptions=TestClass._kernels,
        save_source_as="test_class.c",
        compile=True,
        containing_dir=str(my_folder),
        extra_classes=[TestClass],
        module_name="test_class",
    )
    test_context.kernels.update(kernels)

    tc = TestClass(x=3, y=4)
    assert test_context.kernels.myfun(tc=tc) == 12

    # On a new context, load the kernels and test them,
    # making sure we don't recompile
    cffi_compile = mocker.patch.object(cffi.FFI, "compile")
    fresh_context = xo.ContextCpu()
    loaded_kernels = fresh_context.kernels_from_file(
        "test_class",
        TestClass._kernels,
        str(my_folder),
    )
    fresh_context.kernels.update(loaded_kernels)

    tc = TestClass(x=5, y=7, _context=fresh_context)
    assert fresh_context.kernels.myfun(tc=tc) == 35
    cffi_compile.assert_not_called()

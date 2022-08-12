# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import os
import uuid
import importlib
import sysconfig
import logging

import numpy as np

from .context import (
    XBuffer,
    XContext,
    ModuleNotAvailable,
    available,
    classes_from_kernels,
    sort_classes,
    sources_from_classes,
    _concatenate_sources,
)
from .specialize_source import specialize_source
from .linkedarray import BaseLinkedArray

log = logging.getLogger(__name__)

try:
    import cffi

    _enabled = True
except ImportError:
    log.info("cffi is not installed, ContextCPU will not be available")

    cffi = ModuleNotAvailable(
        message=("cffi is not installed. " "ContextCPU is not available!")
    )
    _enabled = False

try:
    import pyfftw

    pyfftw_available = True
except ImportError:
    log.info("pyfftw not available, will use numpy")
    pyfftw = ModuleNotAvailable(message="pyfftw not available")
    pyfftw_available = False

dtype_dict = {
    "float64": "double",
    "float32": "float",
    "int64": "int64_t",
    "int32": "int32_t",
    "int8": "int8_t",
    "uint64": "uint64_t",
    "uint32": "uint32_t",
}


def dtype2ctype(dtype):
    return dtype_dict[dtype.name]


def nplike_to_numpy(arr):
    return np.array(arr)


def cdef_from_kernel(kernel, pyname=None):
    if kernel.c_name is None:
        kernel.c_name = pyname
    if kernel.ret is not None:
        rettype = kernel.ret.get_c_type()
    else:
        rettype = "void"
    signature = f"{rettype} {kernel.c_name}("
    signature += ",".join(arg.get_c_type() for arg in kernel.args)
    signature += ");"
    return signature

# order of base classes matters as it defines which __setitem__ is used
class LinkedArrayCpu(BaseLinkedArray, np.ndarray):

    @classmethod
    def _build_view(cls, a):
        assert len(a.shape) == 1
        return cls(shape=a.shape, dtype=a.dtype, buffer=a.data, offset=0,
                strides=a.strides, order='C')

class ContextCpu(XContext):
    """

    Creates a CPU Platform object, that allows performing the computations
    on conventional CPUs.

    Returns:
         ContextCpu: platform object.

    """

    @property
    def nplike_array_type(self):
        return np.ndarray

    @property
    def linked_array_type(self):
        return LinkedArrayCpu

    def __init__(self, omp_num_threads=0):
        super().__init__()
        self.omp_num_threads = omp_num_threads

        if self.omp_num_threads > 1:
            raise NotImplementedError("OpenMP parallelization not yet supported!")

    def _make_buffer(self, capacity):
        return BufferNumpy(capacity=capacity, context=self)

    def add_kernels(
        self,
        sources=[],
        kernels=[],
        specialize=True,
        apply_to_source=(),
        save_source_as=None,
        extra_compile_args=["-O3", "-Wno-unused-function"],
        extra_link_args=["-O3"],
        extra_cdef=None,
        extra_classes=[],
        extra_headers=[],
        compile=True
    ):

        """
        Adds user-defined kernels to to the context. The kernel source
        code is provided as a string and/or in source files and must contain
        the kernel names defined in the kernel descriptions.
        Args:
            sources (list): List of source codes that are concatenated before
                compilation. The list can contain strings (raw source code),
                File objects and Path objects.
            kernels (dict): Dictionary with the kernel descriptions
                in the form given by the following examples. The descriptions
                define the kernel names, the type and name of the arguments
                and identify one input argument that defines the number of
                threads to be launched (only on cuda/opencl).
            specialize_code (bool): If True, the code is specialized using
                annotations in the source code. Default is ``True``
            save_source_as (str): Filename for saving the specialized source
                code. Default is ```None```.
        Example:

        .. code-block:: python

            # A simple kernel
            src_code = '''
            /*gpukern*/
            void my_mul(const int n,
                /*gpuglmem*/ const double* x1,
                /*gpuglmem*/ const double* x2,
                /*gpuglmem*/       double* y) {
                int tid = 0 //vectorize_over tid
                y[tid] = x1[tid] * x2[tid];
                //end_vectorize
                }
            '''

            # Prepare description
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
            ctx.add_kernels(
                sources=[src_code],
                kernels=kernel_descriptions,
                save_source_as=None,
            )

            # With a1, a2, b being arrays on the context, the kernel
            # can be called as follows:
            ctx.kernels.my_mul(n=len(a1), x1=a1, x2=a2, y=b)
        """

        classes = classes_from_kernels(kernels)
        classes.update(extra_classes)
        classes = sort_classes(classes)
        cls_sources = sources_from_classes(classes)

        headers = ["#include <stdint.h>"]

        if self.omp_num_threads > 0:
            headers = ["#include <omp.h>"] + headers

        headers += extra_headers

        sources = headers + cls_sources + sources

        source, folders = _concatenate_sources(sources, apply_to_source)

        if specialize:
            if self.omp_num_threads > 0:
                specialize_for = "cpu_openmp"
            else:
                specialize_for = "cpu_serial"
            # included files are searched in the same folders od the src_filed
            specialized_source = specialize_source(
                source,
                specialize_for=specialize_for,
                search_in_folders=list(folders),
            )
        else:
            specialized_source = source

        if save_source_as is not None:
            with open(save_source_as, "w") as fid:
                fid.write(specialized_source)

        if compile:
            ffi_interface = cffi.FFI()

            cdefs = "\n".join(cls._gen_c_decl({}) for cls in classes)

            ffi_interface.cdef(cdefs)

            if extra_cdef is not None:
                ffi_interface.cdef(extra_cdef)

            for pyname, kernel in kernels.items():
                if pyname not in cdefs:  # check if kernel not already declared
                    signature = cdef_from_kernel(kernel, pyname)
                    ffi_interface.cdef(signature)
                    log.debug(f"cffi def {pyname} {signature}")

            if self.omp_num_threads > 0:
                ffi_interface.cdef("void omp_set_num_threads(int);")

            # Generate temp fname
            tempfname = str(uuid.uuid4().hex)

            # Compile
            xtr_compile_args = ["-std=c99"]
            xtr_link_args = ["-std=c99"]
            xtr_compile_args += extra_compile_args
            xtr_link_args += extra_link_args
            if self.omp_num_threads > 0:
                xtr_compile_args.append("-fopenmp")
                xtr_link_args.append("-fopenmp")

            if os.name == 'nt': #windows
                # TODO: to be handled properly
                xtr_compile_args = []
                xtr_link_args = []

            ffi_interface.set_source(
                tempfname,
                specialized_source,
                extra_compile_args=xtr_compile_args,
                extra_link_args=xtr_link_args,
            )

            ffi_interface.compile(verbose=True)

            # build full so filename, something like:
            # 0e14651ea79740119c6e6c24754f935e.cpython-38-x86_64-linux-gnu.so
            suffix = sysconfig.get_config_var("EXT_SUFFIX")
            so_fname = tempfname + suffix

            try:
                # Import the compiled module
                spec = importlib.util.spec_from_file_location(
                    tempfname, os.path.abspath("./" + tempfname + suffix)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if self.omp_num_threads > 0:
                    raise NotImplementedError("OpenMP not supported for now!")
                    self.omp_set_num_threads = module.lib.omp_set_num_threads

                # Get the methods
                for pyname, kernel in kernels.items():
                    self.kernels[pyname] = KernelCpu(
                        function=getattr(module.lib, kernel.c_name),
                        description=kernel,
                        ffi_interface=module.ffi,
                        context=self,
                    )

            finally:
                # Clean temp files
                files_to_remove = [so_fname, tempfname + ".c", tempfname + ".o"]

                for ff in files_to_remove:
                    if os.path.exists(ff):
                        if os.name == 'nt' and ff.endswith('.pyd'):
                            # pyd files are protected on windows
                            continue
                        os.remove(ff)
        else:
            # Only kernel information, but no possibility to call the kernel
            for pyname, kernel in kernels.items():
                self.kernels[pyname] = KernelCpu(
                    function=None,
                    description=kernel,
                    ffi_interface=None,
                    context=self,
                )

        for pyname, kernel in kernels.items():
            self.kernels[pyname].source = source
            self.kernels[pyname].specialized_source = specialized_source
            self.kernels[pyname].description.pyname = pyname # TODO: find better implementation?

    def nparray_to_context_array(self, arr):
        """
        Moves a numpy array to the device memory. No action is performed by
        this function in the CPU context. The method is provided
        so that the CPU context has an identical API to the GPU ones.

        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            numpy.ndarray: The same array (no copy!).

        """
        return arr

    def nparray_from_context_array(self, dev_arr):
        """
        Moves an array to the device to a numpy array. No action is performed by
        this function in the CPU context. The method is provided so that the CPU
        context has an identical API to the GPU ones.

        Args:
            dev_arr (numpy.ndarray): Array to be transferred
        Returns:
            numpy.ndarray: The same array (no copy!)

        """
        return dev_arr

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features. Numpy members should be accessed
        through ``nplike_lib`` to keep compatibility with the other contexts.

        """

        return np

    def synchronize(self):
        """
        Ensures that all computations submitted to the context are completed.
        No action is performed by this function in the CPU context. The method
        is provided so that the CPU context has an identical API to the GPU ones.
        """
        pass

    def zeros(self, *args, **kwargs):
        """
        Allocates an array of zeros on the device. The function has the same
        interface of numpy.zeros"""
        return self.nplike_lib.zeros(*args, **kwargs)

    def plan_FFT(self, data, axes):
        """
        Generate an FFT plan object to be executed on the context.

        Args:
            data (numpy.ndarray): Array having type and shape for which the FFT
                needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            FFTCpu: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = context.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return FFTCpu(data, axes, threads=self.omp_num_threads)

    @property
    def kernels(self):

        """
        Dictionary containing all the kernels that have been imported to the context.
        The syntax ``context.kernels.mykernel`` can also be used.
        """

        return self._kernels


class BufferByteArray(XBuffer):
    def _make_context(self):
        return ContextCpu()

    def _new_buffer(self, capacity):
        return bytearray(capacity)

    def update_from_native(self, offset, source, source_offset, nbytes):
        """Copy data from native buffer into self.buffer starting from offset"""
        self.buffer[offset : offset + nbytes] = source[
            source_offset : source_offset + nbytes
        ]

    def copy_native(self, offset, nbytes):
        """return native data with content at from offset and nbytes"""
        return self.buffer[offset : offset + nbytes].copy()

    def copy_to_native(self, dest, dest_offset, source_offset, nbytes):
        """copy data from self.buffer into dest"""
        dest[dest_offset : dest_offset + nbytes] = self.buffer[
            source_offset : source_offset + nbytes
        ]

    def update_from_buffer(self, offset, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""
        nbytes = len(source)
        self.buffer[offset : offset + nbytes] = source

    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""
        count = np.prod(shape)
        return np.frombuffer(
            self.buffer, dtype=dtype, count=count, offset=offset
        ).reshape(*shape)

    def update_from_nplike(self, offset, dest_dtype, value):
        value = nplike_to_numpy(value)
        if dest_dtype != value.dtype:
            value = value.astype(dtype=dest_dtype)
        self.update_from_native(offset, value.data, 0, value.nbytes)

    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""
        return self.buffer[offset : offset + nbytes]

    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel"""
        return self.buffer[offset : offset + nbytes]


class BufferNumpy(XBuffer):
    def _make_context(self):
        return ContextCpu()

    def _new_buffer(self, capacity):
        return np.zeros(capacity, dtype="int8")

    def update_from_native(self, offset, source, source_offset, nbytes):
        """Copy data from native buffer into self.buffer starting from offset"""
        self.buffer[offset : offset + nbytes] = source[
            source_offset : source_offset + nbytes
        ]

    def copy_native(self, offset, nbytes):
        """return native data with content at from offset and nbytes"""
        return self.buffer[offset : offset + nbytes].copy()

    def copy_to_native(self, dest, dest_offset, source_offset, nbytes):
        """copy data from self.buffer into dest"""
        dest[dest_offset : dest_offset + nbytes] = self.buffer[
            source_offset : source_offset + nbytes
        ]

    def update_from_buffer(self, offset, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""
        nbytes = len(source)
        self.buffer[offset : offset + nbytes] = bytearray(source)

    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""
        count = np.prod(shape)
        # dtype=np.dtype(dtype)
        # return self.buffer[offset:].view(dtype)[:count].reshape(*shape)
        return np.frombuffer(
            self.buffer, dtype=dtype, count=count, offset=offset
        ).reshape(*shape)

    def update_from_nplike(self, offset, dest_dtype, value):
        value = nplike_to_numpy(value)
        if dest_dtype != value.dtype:
            value = value.astype(dtype=dest_dtype)  # make a copy
        src = value.view("int8")
        self.buffer[offset : offset + src.nbytes] = value.flatten().view("int8")

    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""
        return bytearray(self.buffer[offset : offset + nbytes])

    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel"""
        return self.buffer[offset : offset + nbytes]


class KernelCpu:
    def __init__(
        self,
        function,
        description,
        ffi_interface,
        context,
    ):

        self.function = function
        self.description = description
        self.ffi_interface = ffi_interface
        self.context = context

    def to_function_arg(self, arg, value):
        if arg.pointer:
            if hasattr(arg.atype, "_dtype"):  # it is numerical scalar
                if hasattr(value, "dtype"):  # nparray
                    slice_first_elem = value[tuple(value.ndim * [slice(0, 1)])]
                    return self.ffi_interface.cast(
                        dtype2ctype(value.dtype) + "*",
                        self.ffi_interface.from_buffer(slice_first_elem.data),
                    )
                elif hasattr(value, "_shape"):  # xobject array
                    assert isinstance(value._buffer.context, ContextCpu), (
                        f"Incompatible context for argument `{arg.name}`."
                    )
                    return self.ffi_interface.cast(
                        value._c_type + "*",
                        self.ffi_interface.from_buffer(
                            value._buffer.buffer[
                                value._offset + value._data_offset :
                            ]  # fails for pyopencl, cuda
                        ),
                    )
            else:
                raise ValueError(
                    f"Invalid value {value} for argument {arg.name} of kernel {self.description.pyname}"
                )
        else:
            if hasattr(arg.atype, "_dtype"):  # it is numerical scalar
                return arg.atype(value)  # try to return a numpy scalar
            elif hasattr(arg.atype, "_size"):  # it is a compound xobject
                assert isinstance(value._buffer.context, ContextCpu), (
                    f"Incompatible context for argument `{arg.name}`."
                )
                buf = np.frombuffer(value._buffer.buffer, dtype="int8")
                ptr = buf.ctypes.data + value._offset
                return self.ffi_interface.cast(arg.atype._c_type, ptr)
            else:
                raise ValueError(
                    f"Invalid value {value} for argument {arg.name} of kernel {self.description.pyname}"
                )

    def from_function_arg(self, arg, value):
        return value

    @property
    def num_args(self):
        return len(self.description.args)

    def __call__(self, **kwargs):
        if self.function is None:
            raise ValueError(
                f"Kernel {self.description.pyname} is not compiled yet."
            )
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for arg in self.description.args:
            vv = kwargs[arg.name]
            arg_list.append(self.to_function_arg(arg, vv))

        if self.context.omp_num_threads > 0:
            self.context.omp_set_num_threads(self.context.omp_num_threads)

        ret = self.function(*arg_list)

        if self.description.ret is not None:
            return self.from_function_arg(self.description.ret, ret)


class FFTCpu(object):
    def __init__(self, data, axes, threads=0):

        self.axes = axes
        self.threads = threads

        self.use_pyfftw = False
        if threads > 0 and pyfftw_available:
            self.use_pyfftw = True
            self.data = data
            self.data_temp = pyfftw.byte_align(0 * data)
            self.fftw = pyfftw.FFTW(
                data,
                self.data_temp,
                axes=axes,
                threads=threads,
                direction="FFTW_FORWARD",
                flags=("FFTW_MEASURE",),
            )
            self.ifftw = pyfftw.FFTW(
                data,
                self.data_temp,
                axes=axes,
                threads=threads,
                direction="FFTW_BACKWARD",
                flags=("FFTW_MEASURE",),
            )
            print(f"fftw simd_aligned={self.fftw.simd_aligned}")
            print(f"ifftw simd_aligned={self.fftw.simd_aligned}")
        else:
            # I perform one fft to have numpy cache the plan
            _ = np.fft.ifftn(np.fft.fftn(data, axes=axes), axes=axes)

    def transform(self, data):
        """The transform is done inplace"""
        if self.use_pyfftw:
            assert data is self.data
            self.fftw.execute()
            data[:] = self.data_temp[:]
        else:
            data[:] = np.fft.fftn(data, axes=self.axes)[:]

    def itransform(self, data):
        """The transform is done inplace"""
        if self.use_pyfftw:
            assert data is self.data
            self.ifftw.execute()
            data[:] = self.data_temp[:] / self.ifftw.N
        else:
            data[:] = np.fft.ifftn(data, axes=self.axes)[:]


if _enabled:
    available.append(ContextCpu)

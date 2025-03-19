# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import importlib.util
import logging
import os
import sysconfig
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import weakref

from .general import _print

import numpy as np
import scipy as sp

_forbid_compile = False
_suppress_warnings = False

from .context import (
    Kernel,
    ModuleNotAvailable,
    SourceType,
    XBuffer,
    XContext,
    _concatenate_sources,
    available,
    classes_from_kernels,
    sort_classes,
    sources_from_classes,
)
from .linkedarray import BaseLinkedArray
from .specialize_source import specialize_source

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
    "int16": "int16_t",
    "int8": "int8_t",
    "uint64": "uint64_t",
    "uint32": "uint32_t",
    "uint16": "uint16_t",
    "uint8": "uint8_t",
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
        return cls(
            shape=a.shape,
            dtype=a.dtype,
            buffer=a.data,
            offset=0,
            strides=a.strides,
            order="C",
        )

    def copy(self):
        return np.array(self)


def _so_for_module_name(name, containing_dir=".") -> Path:
    # The so file name is something like:
    # 0e14651ea79740119c6e6c24754f935e.cpython-38-x86_64-linux-gnu.so
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    return Path(containing_dir) / (name + suffix)


class ContextCpu(XContext):
    """

    Creates a CPU Platform object, that allows performing the computations
    on conventional CPUs.

    Returns:
         ContextCpu: platform object.

    """

    _cffi_verbose = False
    _compile_kernels_info = True

    @property
    def nplike_array_type(self):
        return np.ndarray

    @property
    def linked_array_type(self):
        return LinkedArrayCpu

    def __init__(self, omp_num_threads=0):
        """
        Create a new CPU context, serial or with parallelization using OpenMP.
        Args:
            omp_num_threads (int | Literal['auto']): Number of threads to be
            used by OpenMP. If 0, no parallelization is used. If 'auto', the
            number of threads is selected automatically by OpenMP.
        """
        super().__init__()
        self.omp_num_threads = omp_num_threads
        if omp_num_threads == 0:
            self.allow_prebuilt_kernels = True

    def __str__(self):
        if not self.openmp_enabled:
            return super().__str__()
        else:
            return f"{type(self).__name__}:{self.omp_num_threads}"

    def _make_buffer(self, capacity):
        return BufferNumpy(capacity=capacity, context=self)

    def add_kernels(
        self,
        sources=None,
        kernels=None,
        specialize=True,
        apply_to_source=(),
        save_source_as=None,
        extra_compile_args: Sequence[str] = (),
        extra_link_args: Sequence[str] = (),
        extra_cdef="",
        extra_classes=(),
        extra_headers=(),
        compile=True,  # noqa
    ):
        """
        Adds user-defined kernels to the context. The kernel source
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
            specialize (bool): If True, the code is specialized using
                annotations in the source code. Default is ``True``
            apply_to_source (List[Callable]): functions to be applied to source
            save_source_as (str): Filename for saving the specialized source
                code. Default is ```None```.
            extra_compile_args: Extra arguments to be passed to the compiler.
            extra_link_args: Extra arguments to be passed to the linker.
            extra_cdef: Extra C definitions to be passed to cffi.
            extra_classes: Extra xobjects classes whose API is needed.
            extra_headers: Extra headers to be added to the source code.
            compile: If True, the source code is compiled. Default is ``True``.
                Otherwise, a dummy kernel is returned, with the source code
                attached.

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
        kernels = kernels or {}
        sources = sources or []

        generated_kernels = self.build_kernels(
            kernel_descriptions=kernels,
            sources=sources,
            specialize=specialize,
            apply_to_source=apply_to_source,
            save_source_as=save_source_as,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            extra_cdef=extra_cdef,
            extra_classes=extra_classes,
            extra_headers=extra_headers,
            compile=compile,
        )
        self.kernels.update(generated_kernels)

    def build_kernels(
        self,
        kernel_descriptions: Dict[str, Kernel],
        module_name: str = None,
        containing_dir=".",
        sources=None,
        specialize=True,
        apply_to_source=(),
        save_source_as=None,
        extra_compile_args=(),
        extra_link_args=(),
        extra_cdef="",
        extra_classes=(),
        extra_headers=(),
        compile=True,  # noqa
    ) -> Dict[Tuple[str, tuple], "KernelCpu"]:
        extra_compile_args += ("-O3", "-Wno-unused-function")
        extra_link_args += ("-O3",)

        # Determine names and paths
        clean_up_so = not module_name
        module_name = module_name or str(uuid.uuid4().hex)
        containing_dir = containing_dir

        classes = list(classes_from_kernels(kernel_descriptions))
        classes += list(extra_classes)
        classes = sort_classes(classes)

        # Update the kernel descriptions with the overriden classes
        cls_for_name = {cls.__name__: cls for cls in classes}
        for kernel_name, kernel in kernel_descriptions.items():
            for arg in kernel.args:
                arg.atype = cls_for_name.get(arg.atype.__name__, arg.atype)

        source, specialized_source = self._build_sources(
            sources=sources,
            classes=classes,
            extra_headers=extra_headers,
            apply_to_source=apply_to_source,
            specialize=specialize,
        )

        if save_source_as is not None:
            Path(containing_dir).mkdir(parents=True, exist_ok=True)
            with (Path(containing_dir) / save_source_as).open("w+") as fid:
                fid.write(specialized_source)

        if compile:
            cdefs = "\n".join(cls._gen_c_decl({}) for cls in classes)
            cdefs += "\n" + extra_cdef

            if _forbid_compile:
                raise RuntimeError("Compilation is forbidden")

            so_file = self.compile_kernel(
                module_name,
                kernel_descriptions,
                cdefs,
                specialized_source,
                extra_compile_args,
                extra_link_args,
                containing_dir=containing_dir,
            )

            try:
                # Import the compiled module
                out_kernels = self.kernels_from_file(
                    module_name,
                    kernel_descriptions,
                    containing_dir=containing_dir,
                )
            finally:
                # Whether loaded successfully or not, delete the so
                if (
                    os.name != "nt" or so_file.suffix != ".pyd"
                ) and clean_up_so:
                    so_file.unlink()
        else:
            # Only kernel information, but no possibility to call the kernel
            out_kernels = {}
            for pyname, kernel in kernel_descriptions.items():
                out_kernels[pyname] = KernelCpu(
                    function=None,
                    description=kernel,
                    ffi_interface=None,
                    context=self,
                )

        for pyname, kernel in out_kernels.items():
            kernel.source = source
            kernel.specialized_source = specialized_source
            # TODO: find better implementation?
            kernel.description.pyname = pyname

        return out_kernels

    def kernels_from_file(
        self,
        module_name: str,
        kernel_descriptions: Dict[str, Kernel],
        containing_dir=".",
    ) -> Dict[Tuple[str, tuple], "KernelCpu"]:
        """
        Import a compiled module `module_name` located in `containing_dir`
        (by default it is the current working directory), and add the kernels
        from the module, as defined in `kernel_descriptions`, to the context.
        Returns the path to the loaded so file.
        """
        module = self._load_kernel_module(
            name=module_name,
            containing_dir=containing_dir,
        )
        out_kernels = {}
        for pyname, kernel_desc in kernel_descriptions.items():
            out_kernels[pyname] = KernelCpu(
                function=getattr(module.lib, kernel_desc.c_name),
                description=kernel_desc,
                ffi_interface=module.ffi,
                context=self,
            )
        return out_kernels

    def compile_kernel(
        self,
        module_name,
        kernel_descriptions,
        cdefs,
        specialized_source,
        extra_compile_args,
        extra_link_args,
        containing_dir=".",
    ) -> Path:
        ffi_interface = cffi.FFI()
        ffi_interface.cdef(cdefs)

        if self._compile_kernels_info:
            _print("Compiling ContextCpu kernels...")

        for pyname, kernel in kernel_descriptions.items():
            # check if kernel not already declared
            if f" {kernel.c_name}(" not in cdefs:
                signature = cdef_from_kernel(kernel, pyname)
                ffi_interface.cdef(signature)
                log.debug(f"cffi def {pyname} {signature}")

        if self.openmp_enabled:
            ffi_interface.cdef("void omp_set_num_threads(int);")
            ffi_interface.cdef("int omp_get_max_threads();")

        # Compile
        xtr_compile_args = ["-std=c99", "-DXO_CONTEXT_CPU"]
        xtr_link_args = ["-std=c99", "-DXO_CONTEXT_CPU"]
        xtr_compile_args += extra_compile_args
        xtr_link_args += extra_link_args

        if self.openmp_enabled:
            xtr_compile_args.append("-fopenmp")
            xtr_link_args.append("-fopenmp")
            xtr_compile_args.append("-DXO_CONTEXT_CPU_OPENMP")
            xtr_link_args.append("-DXO_CONTEXT_CPU_OPENMP")

            # https://mac.r-project.org/openmp/
            # on macos comment the above and uncomment the below flags to compile OpenMP with Xcode clang:
            # xtr_compile_args.append("-Xclang")
            # xtr_compile_args.append("-fopenmp")
            # xtr_link_args.append("-lomp")
        else:
            xtr_compile_args.append("-DXO_CONTEXT_CPU_SERIAL")
            xtr_link_args.append("-DXO_CONTEXT_CPU_SERIAL")

        if os.name == "nt":  # windows
            # TODO: to be handled properly
            xtr_compile_args = []
            xtr_link_args = []

        if _suppress_warnings:
            xtr_compile_args.append("-w")
            xtr_link_args.append("-w")

        ffi_interface.set_source(
            module_name,
            specialized_source,
            extra_compile_args=xtr_compile_args,
            extra_link_args=xtr_link_args,
        )

        try:
            so_file = str(
                _so_for_module_name(module_name, containing_dir).absolute()
            )
            output_file = ffi_interface.compile(
                target=so_file, verbose=self._cffi_verbose
            )
            if self._compile_kernels_info:
                _print("Done compiling ContextCpu kernels.")
            return Path(output_file)
        finally:
            # Clean temp files
            files_to_remove = [
                module_name + ".c",
                module_name + ".o",
            ]

            for ff in files_to_remove:
                if os.path.exists(ff):
                    os.remove(ff)

    def _build_sources(
        self,
        classes: List[type] = None,
        extra_headers: Sequence[SourceType] = None,
        specialize: bool = True,
        sources: Sequence[SourceType] = None,
        apply_to_source: Sequence[Callable] = (),
    ) -> Tuple[str, str]:
        sources = sources or []
        classes = classes or []
        cls_sources = sources_from_classes(classes)

        headers = ["#include <stdint.h>"]

        if self.openmp_enabled:
            headers = ["#include <omp.h>"] + headers

        headers += extra_headers
        sources = headers + cls_sources + sources
        source, folders = _concatenate_sources(sources, apply_to_source)

        if specialize:
            if self.openmp_enabled:
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
        return source, specialized_source

    def _load_kernel_module(
        self,
        name: str,
        containing_dir=".",
    ):
        """
        Load a kernel from a stored shared object file.
        """
        so_path = _so_for_module_name(name, containing_dir)
        # Import the compiled module
        spec = importlib.util.spec_from_file_location(name, so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if self.openmp_enabled:
            self.omp_set_num_threads = module.lib.omp_set_num_threads
            self.omp_get_max_threads = module.lib.omp_get_max_threads

        return module

    @staticmethod
    def cffi_module_for_c_types(c_types, containing_dir="."):
        path = Path(containing_dir)
        for file in path.iterdir():
            if not file.suffix in [".so", ".dylib", ".dll"]:
                continue
            module_name = file.name.split(".")[0]
            spec = importlib.util.spec_from_file_location(
                module_name, str(file)
            )
            module = importlib.util.module_from_spec(spec)

            typedefs = module.ffi.list_types()[0]
            if set(c_types) <= set(typedefs):
                return module_name

        return None

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

    @property
    def splike_lib(self):
        """
        Module containing all the scipy features. Numpy members should be accessed
        through ``splike_lib`` to keep compatibility with the other contexts.

        """

        return sp

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
        try:
            num_threads = self.omp_get_max_threads()
        except AttributeError:
            num_threads = 1
        return FFTCpu(data, axes, threads=num_threads)

    @property
    def kernels(self):
        """
        Dictionary containing all the kernels that have been imported to the context.
        The syntax ``context.kernels.mykernel`` can also be used.
        """

        return self._kernels

    @property
    def openmp_enabled(self):
        return self.omp_num_threads != 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kernels"] = {}
        del state["_buffers"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._buffers = weakref.WeakSet()


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

    def to_native(self, offset, nbytes):
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

    to_nparray = to_nplike

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

    def to_native(self, offset, nbytes):
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

    to_nparray = to_nplike

    def update_from_nplike(self, offset, dest_dtype, value):
        value = nplike_to_numpy(value)
        if dest_dtype != value.dtype:
            value = value.astype(dtype=dest_dtype)  # make a copy
        src = value.view("int8")
        self.buffer[offset : offset + src.nbytes] = value.flatten().view(
            "int8"
        )

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
        self.source = None
        self.specialized_source = None

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
                    assert isinstance(
                        value._buffer.context, ContextCpu
                    ), f"Incompatible context for argument `{arg.name}`."
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
                assert isinstance(
                    value._buffer.context, ContextCpu
                ), f"Incompatible context for argument `{arg.name}`."
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

        if self.context.openmp_enabled:
            if isinstance(self.context.omp_num_threads, int):
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

import os
import uuid
import importlib
import sysconfig

import numpy as np

from .general import Buffer, Context, ModuleNotAvailable, available
from .specialize_source import specialize_source

try:
    import cffi

    _enabled = True
except ImportError:
    print(
        "WARNING:" "cffi is not installed, this platform will not be available"
    )

    cffi = ModuleNotAvailable(
        message=("cffi is not installed. " "this platform is not available!")
    )
    _enabled = False

type_mapping = {np.int32: "int32_t", np.int64: "int64_y", np.float64: "double"}


class ContextCpu(Context):
    """

    Creates a CPU Platform object, that allows performing the computations
    on conventional CPUs.

    Returns:
         ContextCpu: platform object.

    """

    def __init__(self, omp_threads=0):
        super().__init__()
        self.ffi_interface = cffi.FFI()
        self.omp_threads = omp_threads

    def _make_buffer(self, capacity):
        return BufferByteArray(capacity=capacity, context=self)

    def add_kernels(self, src_code="", src_files=[], kernel_descriptions={},
            specialize_code=True, save_src_as='_compiled.c'):

        """
        Adds user-defined kernels to to the context. The kernel source
        code is provided as a string and/or in source files and must contain
        the kernel names defined in the kernel descriptions.

        Args:
            src_code (str): String with the kernel source code. Default: empty
                string.
            src_files (list of strings): paths to files containing the
                source code. Default: empty list.
            kernel_descriptions (dict): Dictionary with the kernel descriptions
                in the form given by the following examples. The decriptions
                define the kernel names, the type and name of the arguments
                and identifies one input argument that defines the number of
                threads to be launched.
            specialize_code (bool): If True, the code is specialized using
                annotations in the source code. Default is ``True``

        Example:

        .. code-block:: python

            src_code = r'''
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid;
                for (tid=0; tid<n; tid++){
                    y[tid] = x1[tid] * x2[tid];
                    }
                }
            '''
            kernel_descriptions = {'my_mul':{
                args':(
                    (('scalar', np.int32),   'n',),
                    (('array',  np.float64), 'x1',),
                    (('array',  np.float64), 'x2',),
                    )
                'num_threads_from_arg': 'n'
                },}

            # Import kernel in context
            context.add_kernels(src_code, kernel_descriptions)

            # With a1 and a2 being arrays on the context, the kernel
            # can be called as follows:
            context.kernels.my_mul(n=len(a1), x1=a1, x2=a2)
        """

        src_content = src_code
        fold_list = []
        for ff in src_files:
            fold_list.append(os.path.dirname(ff))
            with open(ff, "r") as fid:
                src_content += "\n\n" + fid.read()

        if specialize_code:
            # included files are searched in the same folders od the src_filed
            src_content = specialize_source(src_content,
                    specialize_for='cpu_serial', search_in_folders=fold_list)

        if save_src_as is not None:
            with open(save_src_as, 'w') as fid:
                fid.write(src_content)

        ker_names = kernel_descriptions.keys()
        for kk in ker_names:
            signature = f"void {kk}("
            for aa in kernel_descriptions[kk]["args"]:
                tt = aa[0]
                signature += type_mapping[tt[1]]
                signature += {"array": "*", "scalar": ""}[tt[0]]
                signature += ", "
            signature = signature[:-2]  # remove the last comma and space
            signature += ");"

            self.ffi_interface.cdef(signature)

        # Generate temp fname
        tempfname = str(uuid.uuid4().hex)

        # Compile
        extra_compile_args = []
        extra_link_args = []
        if self.omp_threads>0:
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')
        self.ffi_interface.set_source(tempfname, src_content,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args)
        self.ffi_interface.compile(verbose=True)

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

            # Get the methods
            for nn in ker_names:
                kk = getattr(module.lib, nn)
                aa = kernel_descriptions[nn]["args"]
                aa_types, aa_names = zip(*aa)
                self.kernels[nn] = KernelCpu(
                    kernel=kk,
                    arg_names=aa_names,
                    arg_types=aa_types,
                    ffi_interface=self.ffi_interface,
                )
        finally:
            # Clean temp files
            files_to_remove = [so_fname, tempfname + ".c", tempfname + ".o"]
            for ff in files_to_remove:
                if os.path.exists(ff):
                    os.remove(ff)

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
        return FFTCpu(data, axes)

    @property
    def kernels(self):

        """
        Dictionary containing all the kernels that have been imported to the context.
        The syntax ``context.kernels.mykernel`` can also be used.

        Example:

        .. code-block:: python

            src_code = r'''
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid;
                for (tid=0; tid<n; tid++){
                    y[tid] = x1[tid] * x2[tid];
                    }
                }
            '''
            kernel_descriptions = {'my_mul':{
                args':(
                    (('scalar', np.int32),   'n',),
                    (('array',  np.float64), 'x1',),
                    (('array',  np.float64), 'x2',),
                    )
                'num_threads_from_arg': 'n'
                },}

            # Import kernel in context
            context.add_kernels(src_code, kernel_descriptions)

            # With a1 and a2 being arrays on the context, the kernel
            # can be called as follows:
            context.kernels.my_mul(n=len(a1), x1=a1, x2=a2)
            # or as follows:
            context.kernels['my_mul'](n=len(a1), x1=a1, x2=a2)

        """

        return self._kernels


class BufferByteArray(Buffer):
    def _make_context(self):
        return ContextCpu()

    def _new_buffer(self, capacity):
        return bytearray(capacity)

    def copy_to(self, dest):
        dest[:] = self.buffer

    def copy_from(self, source, src_offset, dest_offset, byte_count):
        self.buffer[dest_offset : dest_offset + byte_count] = source[
            src_offset : src_offset + byte_count
        ]

    def write(self, offset, data):
        self.buffer[offset : offset + len(data)] = data

    def read(self, offset, size):
        return self.buffer[offset : offset + size]


# One could implement something like this and chose between Numpy and ByteArr
# when building the context
class NumpyArrayBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class KernelCpu(object):
    def __init__(self, kernel, arg_names, arg_types, ffi_interface):

        assert len(arg_names) == len(arg_types)

        self.kernel = kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.ffi_interface = ffi_interface

        # c_argtypes = []
        # for tt in arg_types:
        #     if tt[0] == "scalar":
        #         if np.issubdtype(tt[1], np.integer):
        #             c_argtypes.append(int)
        #         else:
        #             c_argtypes.append(tt[1])
        #     elif tt[0] == "array":
        #         c_argtypes.append(None)  # Not needed for cppyy
        #     else:
        #         raise ValueError(f"Type {tt} not recognized")
        # self.c_arg_types = c_argtypes

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt in zip(self.arg_names, self.arg_types):
            vv = kwargs[nn]
            if tt[0] == "scalar":
                assert np.isscalar(vv)
                arg_list.append(tt[1](vv))
            elif tt[0] == "array":
                slice_first_elem = vv[tuple(vv.ndim * [slice(0, 1)])]
                arg_list.append(
                    self.ffi_interface.cast(
                        type_mapping[tt[1]] + "*",
                        self.ffi_interface.from_buffer(slice_first_elem),
                    )
                )
            else:
                raise ValueError(f"Type {tt} not recognized")

        event = self.kernel(*arg_list)


class FFTCpu(object):
    def __init__(self, data, axes):

        self.axes = axes

        # I perform one fft to have numpy cache the plan
        _ = np.fft.ifftn(np.fft.fftn(data, axes=axes), axes=axes)

    def transform(self, data):
        """The transform is done inplace"""
        data[:] = np.fft.fftn(data, axes=self.axes)[:]

    def itransform(self, data):
        """The transform is done inplace"""
        data[:] = np.fft.ifftn(data, axes=self.axes)[:]


if _enabled:
    available.append(ContextCpu)

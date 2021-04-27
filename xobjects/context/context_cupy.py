import os

import numpy as np

from .general import XBuffer, XContext, ModuleNotAvailable, available
from .specialize_source import specialize_source


try:
    import cupy
    from cupyx.scipy import fftpack as cufftp

    _enabled = True
except ImportError:
    print("WARNING: cupy is not installed, this context will not be available")
    cupy = ModuleNotAvailable(
        message=("cupy is not installed. " "this context is not available!")
    )
    cufftp = cupy
    _enabled = False


def nplike_to_cupy(arr):
    return cupy.array(arr)


class ContextCupy(XContext):

    """
    Creates a Cupy Context object, that allows performing the computations
    on nVidia GPUs.

    To select device use cupy.Device(<n>).use()

    Args:
        default_block_size (int):  CUDA thread size that is used by default
            for kernel execution in case a block size is not specified
            directly in the kernel object. The default value is 256.
    Returns:
        ContextCupy: context object.

    """

    def __init__(self, default_block_size=256, device=None):

        if device is not None:
            cupy.Device(device).use()

        super().__init__()

        self.default_block_size = default_block_size

    def _make_buffer(self, capacity):
        return BufferCupy(capacity=capacity, context=self)

    def add_kernels(
        self,
        sources,
        kernels,
        specialize=True,
        save_source_as=None,
    ):

        source = []
        source.append('extern "C"{')
        fold_list = set()
        for ss in sources:
            if hasattr(ss, "read"):
                source.append(ss.read())
                fold_list.add(os.path.dirname(ss.name))
            else:
                source.append(ss)
        source.append("}")
        source = "\n".join(source)

        if specialize:
            # included files are searched in the same folders od the src_filed
            source = specialize_source(
                source, specialize_for="cuda", search_in_folders=fold_list
            )

        if save_source_as is not None:
            with open(save_source_as, "w") as fid:
                fid.write(source)

        module = cupy.RawModule(code=source)

        for pyname, kernel in kernels.items():
            if kernel.c_name is None:
                kernel.c_name = pyname

            self.kernels[pyname] = KernelCupy(
                function=module.get_function(kernel.c_name),
                description=kernel,
                block_size=self.default_block_size,
                context=self
            )

    def nparray_to_context_array(self, arr):
        """
        Copies a numpy array to the device memory.

        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            cupy.ndarray:The same array copied to the device.

        """
        dev_arr = cupy.array(arr)
        return dev_arr

    def nparray_from_context_array(self, dev_arr):
        """
        Copies an array to the device to a numpy array.

        Args:
            dev_arr (cupy.ndarray): Array to be transferred.
        Returns:
            numpy.ndarray: The same data copied to a numpy array.

        """
        return dev_arr.get()

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features supported by cupy.
        """
        return cupy

    def synchronize(self):
        """
        Ensures that all computations submitted to the context are completed.
        Equivalent to ``cupy.cuda.stream.get_current_stream().synchronize()``
        """
        cupy.cuda.stream.get_current_stream().synchronize()

    def zeros(self, *args, **kwargs):
        """
        Allocates an array of zeros on the device. The function has the same
        interface of numpy.zeros"""
        return self.nplike_lib.zeros(*args, **kwargs)

    def plan_FFT(
        self,
        data,
        axes,
    ):
        """
        Generates an FFT plan object to be executed on the context.

        Args:
            data (cupy.ndarray): Array having type and shape for which the FFT
                needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            FFTCupy: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = context.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return FFTCupy(self, data, axes)

    @property
    def kernels(self):

        """
        Dictionary containing all the kernels that have been imported to the context.
        The syntax ``context.kernels.mykernel`` can also be used.

        Example:

        .. code-block:: python

            src_code = r'''
            /*gpukern*/
            void my_mul(const int n,
                /*gpuglmem*/ const double* x1,
                /*gpuglmem*/ const double* x2,
                /*gpuglmem*/       double* y) {

                for (int tid=0; tid<n; tid++){ //vectorize_over tid n
                    y[tid] = x1[tid] * x2[tid];
                }//end_vectorize
            }
            '''

            kernel_descriptions = {'my_mul':{
                args':(
                    (('scalar', np.int32),   'n',),
                    (('array',  np.float64), 'x1',),
                    (('array',  np.float64), 'x2',),
                    (('array',  np.float64), 'y',),
                    )
                'num_threads_from_arg': 'n'
                },}

            # Import kernel in context
            context.add_kernels(src_code, kernel_descriptions)

            # With a1, a2, b being arrays on the context, the kernel
            # can be called as follows:
            context.kernels.my_mul(n=len(a1), x1=a1, x2=a2, y=b)

        """

        return self._kernels


class BufferCupy(XBuffer):
    def _make_context(self):
        return ContextCupy()

    def _new_buffer(self, capacity):
        return cupy.zeros(shape=(capacity,), dtype=cupy.uint8)

    def update_from_native(self, offset, source, source_offset, nbytes):
        """Copy data from native buffer into self.buffer starting from offset"""
        self.buffer[offset : offset + nbytes] = source[
            source_offset : source_offset + nbytes
        ]

    def copy_native(self, offset, nbytes):
        """return native data with content at from offset and nbytes"""
        return self.buffer[offset : offset + nbytes]

    def update_from_buffer(self, offset, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""
        nbytes = len(source)
        self.buffer[offset : offset + nbytes] = cupy.array(
            np.frombuffer(source, dtype=np.uint8)
        )

    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""
        nbytes = np.prod(shape) * dtype.itemsize
        return (
            self.buffer[offset : offset + nbytes]
            .asarray(dtype=dtype)
            .reshape(*shape)
        )

    def update_from_nplike(self, offset, dest_dtype, arr):
        if arr.dtype != dest_dtype:
            arr = arr.astype(dest_dtype)
        self.update_from_native(offset, arr.data, 0, arr.nbytes)

    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""
        return self.buffer[offset : offset + nbytes].get().tobytes()

    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel"""
        return self.buffer[offset : offset + nbytes]

    def copy_to(self, dest):
        dest[: len(self.buffer)] = self.buffer

    def copy_from(self, source, src_offset, dest_offset, byte_count):
        self.buffer[dest_offset : dest_offset + byte_count] = source[
            src_offset : src_offset + byte_count
        ]

    def write(self, offset, data):
        self.buffer[offset : offset + len(data)] = cupy.array(
            np.frombuffer(data, dtype=np.uint8)
        )

    def read(self, offset, size):
        return self.buffer[offset : offset + size].get().tobytes()


class KernelCupy(object):
    def __init__(
        self,
        function,
        description,
        block_size,
        context,
    ):

        self.function = function
        self.description = description
        self.block_size = block_size
        self.context = context

    def to_function_arg(self, arg, value):
        if arg.pointer:
            if hasattr(arg.atype, "_dtype"):  # it is numerical scalar
                if hasattr(value, "dtype"):  # nparray
                    assert isinstance(value, cupy.ndarray)
                    return value.data
                elif hasattr(value, "_shape"):  # xobject array
                    raise NotImplementedError
            else:
                raise ValueError(
                    f"Invalid value {value} for argument {arg.name} "
                    f"of kernel {self.description.pyname}"
                )
        else:
            if hasattr(arg.atype, "_dtype"):  # it is numerical scalar
                return arg.atype(value)  # try to return a numpy scalar
            elif hasattr(arg.atype, "_size"):  # it is a compound xobject
                    raise NotImplementedError
            else:
                raise ValueError(
                    f"Invalid value {value} for argument {arg.name} of kernel {self.description.pyname}"
                )

    @property
    def num_args(self):
        return len(self.description.args)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for arg in self.description.args:
            vv = kwargs[arg.name]
            arg_list.append(self.to_function_arg(arg, vv))

        n_threads = kwargs[self.description.n_threads]
        grid_size = int(np.ceil(n_threads / self.block_size))
        self.function((grid_size,), (self.block_size,), arg_list)


class FFTCupy(object):
    def __init__(self, context, data, axes):

        self.context = context
        self.axes = axes

        assert len(data.shape) > max(axes)

        from cupyx.scipy import fftpack as cufftp

        if data.flags.f_contiguous:
            self._ax = [data.ndim-1-aa for aa in axes]
            _dat = data.T
            self.f_contiguous = True
        else:
            self._ax = axes
            _dat = data
            self.f_contiguous = False

        self._fftplan = cufftp.get_fft_plan(
            _dat, axes=self._ax, value_type="C2C"
        )

    def transform(self, data):
        if self.f_contiguous:
            _dat = data.T
        else:
            _dat = data
        _dat[:] = cufftp.fftn(_dat, axes=self._ax, plan=self._fftplan)[:]
        """The transform is done inplace"""

    def itransform(self, data):
        """The transform is done inplace"""
        if self.f_contiguous:
            _dat = data.T
        else:
            _dat = data
        _dat[:] = cufftp.ifftn(_dat, axes=self._ax, plan=self._fftplan)[:]


if _enabled:
    available.append(ContextCupy)

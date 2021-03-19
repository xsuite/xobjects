import weakref

import numpy as np

from .general import Buffer, Context, ModuleNotAvailable

try:
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError:
    print('WARNING: pyopencl is not installed, this context will not be available')
    cl = ModuleNotAvailable(message=('pyopencl is not installed. '
                            'this context is not available!'))
    cla = cl

from ._patch_pyopencl_array import _patch_pyopencl_array


class ContextPyopencl(Context):

    @classmethod
    def print_devices(cls):
        for ip, platform in enumerate(cl.get_platforms()):
            print(f"Context {ip}: {platform.name}")
            for id, device in enumerate(platform.get_devices()):
                print(f"Device {ip}.{id}: {device.name}")

    def __init__(self, device="0.0", patch_pyopencl_array=True):

        """
        Creates a Pyopencl Context object, that allows performing the computations
        on GPUs and CPUs through PyOpenCL.

        Args:
            device (str or Device): The device (CPU or GPU) for the simulation.
            default_kernels (bool): If ``True``, the Xfields defult kernels are
                automatically imported.
            patch_pyopencl_array (bool): If ``True``, the PyOpecCL class is patched to
                allow some operations with non-contiguous arrays.

        Returns:
            ContextPyopencl: context object.

        """

        super().__init__()

        if isinstance(device, str):
            platform, device = map(int, device.split("."))
        else:
            self.device = device
            self.platform = device.platform

        self.platform = cl.get_platforms()[platform]
        self.device = self.platform.get_devices()[device]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        if patch_pyopencl_array:
            _patch_pyopencl_array(cl, cla, self.context)

    def new_buffer(self, capacity=1048576):
        buf = BufferPyopencl(capacity=capacity, context=self)
        self.buffers.append(weakref.finalize(buf, print, "free", repr(buf)))
        return buf

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

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

        Example:

        .. code-block:: python

            src_code = r'''
            __kernel
            void my_mul(const int n, __global const float* x1,
                        __global const float* x2, __global float* y) {
                int tid = get_global_id(0);
                if (tid < n){
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
                'num_threads_from_arg': 'nparticles'
                },}

            # Import kernel in context
            context.add_kernels(src_code, kernel_descriptions)

            # With a1 and a2 being arrays on the context, the kernel
            # can be called as follows:
            context.kernels.my_mul(n=len(a1), x1=a1, x2=a2)
        """

        src_content = src_code
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())

        prg = cl.Program(self.context, src_content).build()

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = getattr(prg, nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = KernelCpu(pyopencl_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                queue=self.queue)

    def nparray_to_context_array(self, arr):

        """
        Copies a numpy array to the device memory.
        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            pyopencl.array.Array:The same array copied to the device.

        """
        dev_arr = cla.to_device(self.queue, arr)
        return dev_arr

    def nparray_from_context_array(self, dev_arr):

        """
        Copies an array to the device to a numpy array.

        Args:
            dev_arr (pyopencl.array.Array): Array to be transferred.
        Returns:
            numpy.ndarray: The same data copied to a numpy array.

        """
        return dev_arr.get()

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features supported by PyOpenCL (optionally
        with patches to operate with non-contiguous arrays).
        """
        return cla

    def synchronize(self):
        """
        Ensures that all computations submitted to the context are completed.
        No action is performed by this function in the Pyopencl context. The method
        is provided so that the Pyopencl context has an identical API to the Cupy one.
        """
        pass

    def zeros(self, *args, **kwargs):
        """
        Allocates an array of zeros on the device. The function has the same
        interface of numpy.zeros"""
        return self.nplike_lib.zeros(queue=self.queue, *args, **kwargs)

    def plan_FFT(self, data, axes, wait_on_call=True):
        """
        Generates an FFT plan object to be executed on the context.

        Args:
            data (pyopencl.array.Array): Array having type and shape for which
                the FFT needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            FFTPyopencl: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = context.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return FFTPyopencl(self, data, axes, wait_on_call)

    @property
    def kernels(self):

        """
        Dictionary containing all the kernels that have been imported to the context.
        The syntax ``context.kernels.mykernel`` can also be used.

        Example:

        .. code-block:: python

            src_code = r'''
            __kernel
            void my_mul(const int n, __global const float* x1,
                        __global const float* x2, __global float* y) {
                int tid = get_global_id(0);
                if (tid < n){
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
                'num_threads_from_arg': 'nparticles'
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


class BufferPyopencl(Buffer):


    _DefaultContext = ContextPyopencl

    def _new_buffer(self, capacity):
        return cl.Buffer(
            self.context.context, cl.mem_flags.READ_WRITE, capacity
        )

    def copy_to(self, dest):
        # Does not pass through cpu if it can
        # dest: python object that uses buffer protocol or opencl buffer
        cl.enqueue_copy(self.context.queue, dest, self.buffer)

    def copy_from(self, source, src_offset, dest_offset, byte_count):
        # Does not pass through cpu if it can
        # source: python object that uses buffer protocol or opencl buffer
        cl.enqueue_copy(
            self.context.queue, self.buffer, source, src_offset, dest_offset, byte_count
        )

    def write(self, offset, data):
        # From python object on cpu
        cl.enqueue_copy(
            self.context.queue, self.buffer, data, device_offset=offset
        )

    def read(self, offset, size):
        # To python object on cpu
        data = bytearray(size)
        cl.enqueue_copy(
            self.context.queue, data, self.buffer, device_offset=offset
        )
        return data


class KernelCpu(object):

    def __init__(self, pyopencl_kernel, arg_names, arg_types,
                 num_threads_from_arg, queue,
                 wait_on_call=True):

        assert (len(arg_names) == len(arg_types) == pyopencl_kernel.num_args)
        assert num_threads_from_arg in arg_names

        self.pyopencl_kernel = pyopencl_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.queue = queue
        self.wait_on_call = wait_on_call

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt in zip(self.arg_names, self.arg_types):
            vv = kwargs[nn]
            if tt[0] == 'scalar':
                assert np.isscalar(vv)
                arg_list.append(tt[1](vv))
            elif tt[0] == 'array':
                assert isinstance(vv, cla.Array)
                assert vv.context == self.pyopencl_kernel.context
                arg_list.append(vv.base_data[vv.offset:])
            else:
                raise ValueError(f'Type {tt} not recognized')

        event = self.pyopencl_kernel(self.queue,
                (kwargs[self.num_threads_from_arg],),
                None, *arg_list)

        if self.wait_on_call:
            event.wait()

        return event

class FFTPyopencl(object):
    def __init__(self, context, data, axes, wait_on_call=True):

        self.context = context
        self.axes = axes
        self.wait_on_call = wait_on_call

        assert len(data.shape) > max(axes)

        # Check internal dimensions are powers of two
        for ii in axes[:-1]:
            nn = data.shape[ii]
            frac_part, _ = np.modf(np.log(nn)/np.log(2))
            assert np.isclose(frac_part, 0) , ('PyOpenCL FFT requires'
                    ' all dimensions apart from the last to be powers of two!')

        import gpyfft
        self._fftobj = gpyfft.fft.FFT(context.context,
                context.queue, data, axes=axes)

    def transform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data)
        if self.wait_on_call:
            event.wait()
        return event

    def itransform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data, forward=False)
        if self.wait_on_call:
            event.wait()
        return event

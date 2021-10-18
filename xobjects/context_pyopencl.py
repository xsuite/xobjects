import os
import logging

import numpy as np

from .context import Arg

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

log = logging.getLogger(__name__)

try:
    import pyopencl as cl
    import pyopencl.array as cla

    _enabled = True
except ImportError:
    log.info(
        "pyopencl is not installed, ContextPyopencl will not be available"
    )
    cl = ModuleNotAvailable(
        message=(
            "pyopencl is not installed. " "ContextPyopencl is not available!"
        )
    )
    cl.Buffer = cl
    cla = cl
    _enabled = False

from ._patch_pyopencl_array import _patch_pyopencl_array


openclheader = [
    """\
#ifndef XOBJ_STDINT
typedef long int64_t;
typedef char int8_t;
typedef unsigned int uint32_t;
#endif"""
]

class ContextPyopencl(XContext):
    @classmethod
    def get_devices(cls):
        out = []
        for ip, platform in enumerate(cl.get_platforms()):
            for id, device in enumerate(platform.get_devices()):
                out.append(f"{ip}.{id}")
        return out

    @classmethod
    def print_devices(cls):
        for ip, platform in enumerate(cl.get_platforms()):
            print(f"Context {ip}: {platform.name}")
            for id, device in enumerate(platform.get_devices()):
                print(f"Device {ip}.{id}: {device.name}")

    def __init__(
        self,
        device=None,
        patch_pyopencl_array=True,
        minimum_alignment=None,
        enable_profiling=False,
    ):

        """
        Creates a Pyopencl Context object, that allows performing the computations
        on GPUs and CPUs through PyOpenCL.

        Args:
            device (str or Device): The device (CPU or GPU) for the simulation.
            default_kernels (bool): If ``True``, the Xfields defult kernels are
                automatically imported.
            patch_pyopencl_array (bool): If ``True``, the PyOpecCL class is patched to
                allow some operations with non-contiguous arrays.
            specialize_code (bool): If True, the code is specialized using
                annotations in the source code. Default is ``True``

        Returns:
            ContextPyopencl: context object.

        """

        super().__init__()

        if enable_profiling:
            _queue_prop = cl.command_queue_properties.PROFILING_ENABLE
            self.profiling_enabled = True
        else:
            _queue_prop = 0
            self.profiling_enabled = False

        # TODO assume one device only
        if device is None:
            self.context = cl.create_some_context(interactive=False)
            self.device = self.context.devices[0]
            self.platform = self.device.platform
        else:
            if isinstance(device, str):
                platform, device = map(int, device.split("."))
            else:
                self.device = device
                self.platform = device.platform

            self.platform = cl.get_platforms()[platform]
            self.device = self.platform.get_devices()[device]
            self.context = cl.Context([self.device])

        self.queue = cl.CommandQueue(self.context, properties=_queue_prop)

        if patch_pyopencl_array:
            _patch_pyopencl_array(cl, cla, self.context)

        if minimum_alignment is None:
            minimum_alignment = self.find_minimum_alignment()
        self.minimum_alignment = minimum_alignment

    def _make_buffer(self, capacity):
        return BufferPyopencl(capacity=capacity, context=self)

    def find_minimum_alignment(self):
        buff = self.new_buffer()
        i = 1
        found = False
        while i < 2 ** 16:
            try:
                buff.buffer[i:]
                found = True
                break
            except cl._cl.RuntimeError:
                pass
            i += 1
        if not found:
            raise RuntimeError(
                "Impossible to find minimum alignment on Pyopencl context"
            )
        return i

    def add_kernels(
        self,
        sources=[],
        kernels=[],
        specialize=True,
        save_source_as=None,
        extra_cdef=None,
        extra_classes=[],
        extra_headers=[],
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

        headers = openclheader + extra_headers

        sources = headers + cls_sources + sources

        source, folders = _concatenate_sources(sources)

        if specialize:
            # included files are searched in the same folders od the src_filed
            specialized_source = specialize_source(
                source, specialize_for="opencl", search_in_folders=folders
            )
        else:
            specialized_source = source

        if save_source_as is not None:
            with open(save_source_as, "w") as fid:
                fid.write(specialized_source)

        prg = cl.Program(self.context, specialized_source).build()

        for pyname, kernel in kernels.items():
            if kernel.c_name is None:
                kernel.c_name = pyname

            self.kernels[pyname] = KernelPyopencl(
                function=getattr(prg, kernel.c_name),
                description=kernel,
                context=self,
            )

            self.kernels[pyname].source = source
            self.kernels[pyname].specialized_source = specialized_source

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
        return self.nplike_lib.zeros(self.queue, *args, **kwargs)

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
        """

        return self._kernels


class BufferPyopencl(XBuffer):
    def _make_context(self):
        return ContextPyopencl()

    def _new_buffer(self, capacity):
        return cl.Buffer(
            self.context.context, cl.mem_flags.READ_WRITE, capacity
        )

    def copy_from(self, source, src_offset, dest_offset, byte_count):
        # Does not pass through cpu if it can
        # source: python object that uses buffer protocol or opencl buffer
        cl.enqueue_copy(
            self.context.queue,
            self.buffer,
            source,
            src_offset,
            dest_offset,
            byte_count,
        )

    def write(self, offset, data):
        # From python object with buffer interface on cpu
        log.debug(f"write {self} {offset} {data}")
        cl.enqueue_copy(
            self.context.queue, self.buffer, data, device_offset=offset
        )

    def read(self, offset, size):
        # To bytearray on cpu
        data = bytearray(size)
        cl.enqueue_copy(
            self.context.queue, data, self.buffer, device_offset=offset
        )
        return data

    def update_from_native(
        self, offset: int, source: cl.Buffer, source_offset: int, nbytes: int
    ):
        """Copy data from native buffer into self.buffer starting from offset"""
        cl.enqueue_copy(
            self.context.queue,
            self.buffer,
            source,
            src_offset=source_offset,
            dest_offset=offset,
            byte_count=nbytes,
        )

    def copy_native(self, offset: int, nbytes: int):
        """return native data with content at from offset and nbytes"""
        buff = self._new_buffer(nbytes)
        cl.enqueue_copy(
            queue=self.context.queue,
            dest=buff,
            src=self.buffer,
            src_offset=offset,
            byte_count=nbytes,
        )
        return buff

    def copy_to_native(
        self, dest, dest_offset, source_offset: int, nbytes: int
    ):
        """return native data with content at from offset and nbytes"""
        cl.enqueue_copy(
            queue=self.context.queue,
            dest=dest,
            src=self.buffer,
            src_offset=source_offset,
            byte_count=nbytes,
        )

    def update_from_buffer(self, offset: int, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""
        cl.enqueue_copy(
            queue=self.context.queue,
            dest=self.buffer,
            src=source,  # nbytes taken from min(len(source),len(buffer))
            device_offset=offset,
        )

    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""
        return cl.array.Array(
            self.context.queue,
            data=self.buffer,
            offset=offset,
            dtype=dtype,
            shape=tuple(shape),
        )

    def update_from_nplike(self, offset, dest_dtype, arr):
        if arr.dtype != dest_dtype:
            arr = arr.astype(dest_dtype)
        self.update_from_native(offset, arr.base_data, arr.offset, arr.nbytes)

    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""
        data = bytearray(nbytes)
        cl.enqueue_copy(
            queue=self.context.queue,
            dest=data,  # nbytes taken from min(len(data),len(buffer))
            src=self.buffer,
            device_offset=offset,
        )
        return data

    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel

        Can fail if offset is not a multiple of self.alignment

        """
        return self.buffer[offset : offset + nbytes]


class ProfileResultPyopencl(object):
    def __init__(self):
        self.queued = 0
        self.submit = 0
        self.start = 0
        self.end = 0

    def update(self, event):
        self.submit = event.profile.submit
        self.queued = event.profile.queued
        self.start = event.profile.start
        self.end = event.profile.end

    @property
    def execution_time_ns(self):
        assert self.end >= self.start
        return self.end - self.start

    @property
    def execution_time(self):
        return float(1e-9) * self.execution_time_ns

    @property
    def time_since_submit_ns(self):
        assert self.end >= self.submit
        return self.end - self.submit

    @property
    def time_since_submit(self):
        return float(1e-9) * self.time_since_submit_ns

    @property
    def time_since_queued_ns(self):
        assert self.end >= self.queued
        return self.end - self.queued

    @property
    def time_since_queued(self):
        return float(1e-9) * self.time_since_queued_ns

class KernelPyopenclBase(object):
    @staticmethod
    def get_work_group_info( kernel, cl_dev ):
        assert isinstance( kernel, cl.Kernel )
        assert isinstance( cl_dev, cl.Device )
        return {
            'CL_KERNEL_WORK_GROUP_SIZE': kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, cl_dev ),
            'CL_KERNEL_COMPILE_WORK_GROUP_SIZE': kernel.get_work_group_info(
                cl.kernel_work_group_info.COMPILE_WORK_GROUP_SIZE, cl_dev ),
            'CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE': kernel.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                cl_dev ),
            'CL_KERNEL_LOCAL_MEM_SIZE': kernel.get_work_group_info(
                cl.kernel_work_group_info.LOCAL_MEM_SIZE, cl_dev ),
            'CL_KERNEL_PRIVATE_MEM_SIZE': kernel.get_work_group_info(
                cl.kernel_work_group_info.LOCAL_MEM_SIZE, cl_dev ),
            }

    @staticmethod
    def get_kernel_arg_address_qualifier_infos( cl_kernel, build_options=None ):
        assert isinstance( cl_kernel, cl.Kernel )
        assert build_options is None or isinstance( build_options, str )

        if build_options is None:
            cl_program = cl_kernel.get_info( cl.kernel_info.PROGRAM )
            cl_devices = cl_program.get_info( cl.program_info.DEVICES )
            assert len( cl_devices ) == 1 #TODO: fix for multi-device builds
            cl_dev = cl_devices[ 0 ]
            build_options = cl_program.get_build_info(
                cl_dev, cl.program_build_info.OPTIONS )

        if not isinstance( build_options, str ):
            raise ValueError( "no legal build_options string available" )

        if build_options.find( "-cl-kernel-arg-info" ) == -1:
            raise RuntimeError( "Program appears to have been built "
                "without support for kernel argument info retrieval" )

        num_args = cl_kernel.get_info( cl.kernel_info.NUM_ARGS )

        qual_arg_indices = {
            'CL_KERNEL_ARG_ADDRESS_GLOBAL': [],
            'CL_KERNEL_ARG_ADDRESS_LOCAL': [],
            'CL_KERNEL_ARG_ADDRESS_CONSTANT': [],
            'CL_KERNEL_ARG_ADDRESS_PRIVATE': [],
        }

        for idx in range( num_args ):
            xs_qualifier = cl_kernel.get_arg_info(
                idx, cl.kernel_arg_info.ADDRESS_QUALIFIER )
            if xs_qualifier == cl.kernel_arg_address_qualifier.GLOBAL:
                qual_arg_indices[ "CL_KERNEL_ARG_ADDRESS_GLOBAL" ].append( idx )
            elif xs_qualifier == cl.kernel_arg_address_qualifier.LOCAL:
                qual_arg_indices[ "CL_KERNEL_ARG_ADDRESS_LOCAL" ].append( idx )
            elif xs_qualifier == cl.kernel_arg_address_qualifier.CONSTANT:
                qual_arg_indices[ "CL_KERNEL_ARG_ADDRESS_CONSTANT" ].append( idx )
            else:
                assert xs_qualifier == cl.kernel_arg_address_qualifier.PRIVATE
                qual_arg_indices[ "CL_KERNEL_ARG_ADDRESS_PRIVATE" ].append( idx )

        assert len( qual_arg_indices[ 'CL_KERNEL_ARG_ADDRESS_GLOBAL' ] ) + \
            len( qual_arg_indices[ 'CL_KERNEL_ARG_ADDRESS_LOCAL' ] ) + \
            len( qual_arg_indices[ 'CL_KERNEL_ARG_ADDRESS_CONSTANT' ] ) + \
            len( qual_arg_indices[ 'CL_KERNEL_ARG_ADDRESS_PRIVATE' ] ) == \
            num_args

        return qual_arg_indices


    def __init__(
        self,
        function,
        description,
        context,
        build_options=None
    ):
        self.function = function
        self.description = description
        self.context = context

        kinfo = KernelPyopenclBase.get_work_group_info(
            self.function, self.context.device)

        try:
            arg_info = KernelPyopenclBase.get_kernel_arg_address_qualifier_infos(
                function, build_options )
        except RuntimeError:
            arg_info = {}

        self._global_arg_indices = arg_info.get(
            'CL_KERNEL_ARG_ADDRESS_GLOBAL', None )

        self._local_arg_indices = arg_info.get(
            'CL_KERNEL_ARG_ADDRESS_LOCAL', None )

        self._constant_arg_indices = arg_info.get(
            'CL_KERNEL_ARG_ADDRESS_CONSTANT', None )

        self._private_arg_indices = arg_info.get(
            'CL_KERNEL_ARG_ADDRESS_PRIVATE', None )

        #TODO: Generalise to grid dimensions > 1
        self._grid_dim = 1
        self._global_offset = None
        self._allow_empty_ndrange = False

        self._max_work_group_size = kinfo.get(
            'CL_KERNEL_WORK_GROUP_SIZE', 0 )

        self._pref_work_group_size_multiple = kinfo.get(
            'CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE', 1 )

        self._local_mem_size = kinfo.get( 'CL_KERNEL_LOCAL_MEM_SIZE', 0 )
        self._private_mem_size = kinfo.get( 'CL_KERNEL_PRIVATE_MEM_SIZE', 0 )

        compile_wg_size = tuple( kinfo.get(
            'CL_KERNEL_COMPILE_WORK_GROUP_SIZE', (0,0,0) ) )

        # As mandated by the OpenCL standard:
        assert len( compile_wg_size ) == 3

        #TODO: Generalise to grid dimensions > 1
        assert ( all( compile_wg_size[ ii ] == 0 for ii in range( 3 ) ) or
                ( compile_wg_size[ 0 ] > 0 and
                  compile_wg_size[ 1 ] == 1 and
                  compile_wg_size[ 2 ] == 1 ) )
        self._compile_work_group_size = ( compile_wg_size[ 0 ], ) # 1D !!!

    @property
    def grid_dimension(self):
        assert self._grid_dim >= 1 and self._grid_dim <= 3
        return self._grid_dim

    @property
    def global_offset(self):
        return self._global_offset

    @property
    def allow_empty_ndrange(self):
        return self._allow_empty_ndrange

    @property
    def has_compile_work_group_size(self):
        assert self._grid_dim == 1, "TODO: update for grid dimension > 1"
        assert len( self._compile_work_group_size ) >= 1
        return self._compile_work_group_size[ 0 ] > 0

    @property
    def compile_work_group_num_work_items(self):
        n_work_items = 1
        for wg_size in self._compile_work_group_size:
            n_work_items *= wg_size
        return n_work_items

    @property
    def compile_work_group_size(self):
        assert self.grid_dimension == len( self._compile_work_group_size )
        return self._compile_work_group_size

    @property
    def cl_compile_work_group_size(self):
        cmp_wg_size = [ 0, 0, 0 ]
        for ii in range( min( 3, self._grid_dim ) ):
            cmp_wg_size[ ii ] = self._compile_work_group_size[ ii ]
        return tuple( cmp_wg_size )

    @property
    def max_work_group_size(self):
        return self._max_work_group_size

    @property
    def preferred_work_group_size_multiple(self):
        return self._pref_work_group_size_multiple

    @property
    def private_mem_size(self):
        return self._private_mem_size

    @property
    def local_mem_size(self):
        return self._local_mem_size

    @property
    def num_args(self):
        return len(self.description.args)

    @property
    def num_global_mem_args(self):
        assert self._global_arg_indices is not None
        return len( self._global_arg_indices )

    @property
    def num_local_mem_args(self):
        assert self._local_arg_indices is not None
        return len( self._local_arg_indices )

    @property
    def num_constant_mem_args(self):
        assert self._constant_arg_indices is not None
        return len( self._constant_arg_indices )

    @property
    def local_mem_args(self):
        if self._local_arg_indices is None:
            raise RuntimeError( "not properly initialised global arg indices" )
        return self._local_arg_indices

    def __call__(self, **kwargs):
        raise RuntimeError( "KernelPyopenclBase does not support launching " +
            "kernels, use the derived KernelPyopencl instead!" )

class LocalMemPyopenclArg(Arg):
    @staticmethod
    def calc_local_mem_num_bytes(
        n_work_items_per_wg,
        n_bytes_per_work_item,
        n_bytes_per_wg = 0 ):
        assert n_work_items_per_wg >= 0
        assert n_bytes_per_work_item >= 0
        assert n_bytes_per_wg >= 0
        return n_work_items_per_wg * n_bytes_per_work_item + n_bytes_per_wg

    def __init__(
        self,
        atype,
        name=None,
        num_bytes_per_thread=0,
        num_bytes_common=0,
        **kwargs,
    ):
        if not kwargs.get("pointer", False):
            kwargs["pointer"] = True
        super().__init__(atype, name=name, **kwargs)

        self._loc_mem_num_bytes = 0
        self._cl_loc_mem_obj = None
        self._cl_kernel = None
        self._arg_index = -1

        self._num_bytes_per_work_item = max( 0, num_bytes_per_thread )
        self._num_bytes_per_wg = max( 0, num_bytes_common )

    @property
    def local_memory_obj(self):
        return self._cl_loc_mem_obj

    @property
    def local_memory_num_bytes(self):
        return self._loc_mem_num_bytes

    @property
    def num_bytes_per_thread(self):
        return self._num_bytes_per_work_item

    @property
    def num_bytes_common(self):
        return self._num_bytes_per_wg

    @property
    def is_assigned_to_any_kernel(self):
        assert self._cl_kernel is None or isinstance( self._cl_kernel, cl.Kernel )
        assert self._cl_loc_mem_obj is None or \
            isinstance( self._cl_loc_mem_obj, cl.LocalMemory )
        return self._cl_kernel is not None and self._cl_loc_mem_obj is not None and \
            self._arg_index >= 0

    def is_assigned_to_kernel( self, kernel ):
        result = False
        if self.is_assigned_to_any_kernel:
            if isinstance( kernel, KernelPyopenclBase ):
                result = bool( self._cl_kernel == kernel.function and
                    kernel.num_args > self._arg_index and
                    isinstance( kernel.description.args[ self._arg_index ],
                            LocalMemPyopenclArg ) and
                    kernel.description.args[ self._arg_index ] == self )
            elif isinstance( kernel, cl.Kernel ):
                result = bool( self._cl_kernel == kernel )
        return result

    def assign_to_pyopencl_kernel( self, cl_kernel, arg_index, work_group_size ):
        assert self._cl_kernel is None
        assert self._cl_loc_mem_obj is None
        assert self._arg_index < 0
        assert isinstance( cl_kernel, cl.Kernel )
        loc_mem_num_bytes = self.calc_local_mem_num_bytes(
            work_group_size, self._num_bytes_per_work_item, self._num_bytes_per_wg )
        loc_mem_obj = cl.LocalMemory( loc_mem_num_bytes )

        self._loc_mem_num_bytes = loc_mem_num_bytes
        self._cl_kernel = cl_kernel
        self._arg_index = arg_index
        self._cl_loc_mem_obj = loc_mem_obj


    def assign_to_kernel( self, kernel, arg_index ):
        if not self.is_assigned_to_kernel( kernel ):
            if not self.is_assigned_to_any_kernel:
                assert isinstance( kernel, KernelPyopenclBase )
                assert arg_index < kernel.num_args
                assert self == kernel.description.args[ arg_index ]
                self.assign_to_pyopencl_kernel(
                    kernel.function, arg_index,
                        kernel.local_work_size_num_work_items )
            else:
                raise ValueError( "Can't assign argument to a new kernel, is " +
                    "already assigned to a different kernel" )


class KernelPyopencl(KernelPyopenclBase):
    def __init__(
        self,
        function,
        description,
        context,
        wait_on_call=True,
        profile=True,
        build_options=None
    ):
        super().__init__( function, description, context,
                         build_options=build_options )
        self.wait_on_call = wait_on_call

        if self._global_arg_indices is None:
            # Could not get from OpenCL meta data -> try to guess them from description
            self._global_arg_indices = []
            self._local_arg_indices = []
            self._constant_arg_indices = []
            self._private_arg_indices = []
            for idx, arg in enumerate( description.args ):
                if arg.pointer:
                    #TODO: figure out a way to determine __constant memory
                    self._global_arg_indices.append( idx )
                elif isinstance( arg, LocalMemPyopenclArg ):
                    self._local_arg_indices.append( idx )
                else:
                    self._private_arg_indices.append( idx )

        self._arg_name_to_index = dict( {
            arg.name: idx for idx, arg in enumerate( self.description.args ) } )
        # Verify that there were no duplicate arg.name entries in the list:
        assert len( self._arg_name_to_index ) == len( self.description.args )

        # Ensure that some "reserved" keywords of Pyopencl are not
        # used as argument names
        assert "wait_for" not in self._arg_name_to_index, "illegal arg name wait_for"
        assert "g_times_l" not in self._arg_name_to_index, "illegal arg name g_times_l"

        if self.has_compile_work_group_size:
            assert self.grid_dimension == 1
            #TODO: generalise for grid dimensions > 1
            self._local_work_size = ( self.compile_work_group_num_work_items, )
        elif self.num_local_mem_args > 0:
            assert self.grid_dimension == 1
            #TODO: generalise for grid dimensions > 1
            self._local_work_size = ( self.max_work_group_size, )
        else:
            self._local_work_size = None

        if bool(context.profiling_enabled and profile):
            self.profiling_enabled = True
            self.last_profile = ProfileResultPyopencl()
        else:
            self.profiling_enabled = False
            self.last_profile = None

    @property
    def local_mem_arg_indices(self):
        return self._local_arg_indices

    @property
    def local_work_size(self):
        assert self._local_work_size is None or \
            len( self._local_work_size ) == self.grid_dimension
        return self._local_work_size

    def set_local_work_size( self, new_loc_work_size, strict_multiples=True ):
        if new_loc_work_size is None:
            if self.num_local_mem_args > 0:
                raise ValueError( "Can't set local work size to the default " +
                    "becuase local memory arguments are used" )
            self._local_work_size = None
            return

        #TODO: Generalise for grid dimensions > 1
        assert self.grid_dimension == 1
        loc_work_size = None
        try:
            loc_work_size_iter = iter( new_loc_work_size )
            if len( new_loc_work_size ) > 0:
                loc_work_size = ( new_loc_work_size[ 0 ], )
        except TypeError:
            pass
        if loc_work_size is None:
            loc_work_size = ( int( new_loc_work_size ), )
        assert loc_work_size[ 0 ] >= 0
        assert self.preferred_work_group_size_multiple > 0
        if loc_work_size[ 0 ] > self.max_work_group_size:
            raise ValueError( f"local work size {new_loc_work_size} exceeds " +
                f"maximum allowed work group size of {self.max_work_group_size}" )
        if self.preferred_work_group_size_multiple > 1 and strict_multiples and \
            loc_work_size[ 0 ] % self.preferred_work_group_size_multiple != 0:
            raise ValueError( f"provided local work size {loc_work_size} not " +
                "divisible by the preferred work group size multiplier " +
                    f"{self.preferred_work_group_size_multiple}" )
        self._local_work_size = loc_work_size

    @property
    def local_work_size_num_work_items(self):
        n_work_items = 1
        if self._local_work_size is not None:
            assert len( self._local_work_size ) == self.grid_dimension
            for ii in range( self.grid_dimension ):
                n_work_items *= self._local_work_size[ ii ]
        return n_work_items

    def calc_global_work_size( self, min_num_work_items ):
        #TODO: Generalise for grid dimensions > 1
        assert self.grid_dimension == 1
        n_work_items_multiple = self.preferred_work_group_size_multiple
        assert n_work_items_multiple > 0
        n_work_groups = min_num_work_items // n_work_items_multiple
        n_work_items = n_work_groups *  n_work_items_multiple
        if n_work_items < min_num_work_items:
            n_work_items += n_work_items_multiple
        return ( n_work_items, )

    def to_function_arg(self, arg, value):
        if arg.pointer:
            if hasattr(arg.atype, "_dtype"):  # it is numerical scalar
                if isinstance(value, cl.Buffer):
                    return value
                elif hasattr(value, "dtype"):  # nparray
                    assert isinstance(value, cla.Array)
                    return value.base_data[value.offset :]
                elif hasattr(value, "_shape"):  # xobject array
                    raise NotImplementedError
                else:
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
                return value._buffer.buffer[value._offset :]
            else:
                raise ValueError(
                    f"Invalid value {value} for argument {arg.name} of kernel {self.description.pyname}"
                )

    def update_last_profile(self, event):
        if self.profiling_enabled:
            assert self.last_profile is not None
            self.last_profile.update(event)

    def __call__(self, **kwargs):
        arg_list = []
        for idx, arg in enumerate( self.description.args ):
            if not isinstance( arg, LocalMemPyopenclArg ):
                if arg.name not in kwargs:
                    raise ValueError( "Mandatory kernel argument " +
                        f"#{idx} \"{arg.name}\" not provided." )
                vv = kwargs[arg.name]
                arg_list.append(self.to_function_arg(arg, vv))
            else:
                pdb.set_trace()
                lmem_obj = kwargs.get( arg.name, None )
                if lmem_obj is None:
                    arg.assign_to_kernel( self, idx )
                    arg_list.append( arg.local_memory_obj )
                else:
                    assert isinstance( lmem_obj, cl.LocalMemory )
                    arg_list.append( lmem_obj )
        assert self.num_args == len( arg_list )

        if isinstance(self.description.n_threads, str):
            n_threads = kwargs[self.description.n_threads]
        else:
            n_threads = self.description.n_threads

        event = self.function(
            self.context.queue,
            self.calc_global_work_size( n_threads ),
            self.local_work_size,
            *arg_list,
            global_offset=self.global_offset,
            wait_for = kwargs.get( 'wait_for', None ),
            allow_empty_ndrange = self.allow_empty_ndrange )

        if self.wait_on_call:
            event.wait()
            if self.profiling_enabled:
                self.last_profile.update(event)

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
            frac_part, _ = np.modf(np.log(nn) / np.log(2))
            assert np.isclose(frac_part, 0), (
                "PyOpenCL FFT requires"
                " all dimensions apart from the last to be powers of two!"
            )

        import gpyfft

        self._fftobj = gpyfft.fft.FFT(
            context.context, context.queue, data, axes=axes
        )

    def transform(self, data):
        """The transform is done inplace"""

        (event,) = self._fftobj.enqueue_arrays(data)
        if self.wait_on_call:
            event.wait()
        return event

    def itransform(self, data):
        """The transform is done inplace"""

        (event,) = self._fftobj.enqueue_arrays(data, forward=False)
        if self.wait_on_call:
            event.wait()
        return event


if _enabled:
    available.append(ContextPyopencl)

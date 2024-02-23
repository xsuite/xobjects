# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2022.                   #
# ########################################### #

import logging

import numpy as np
from typing import List, Dict, Tuple

from .context import (
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
    import pyopencl as cl
    import pyopencl.array as cla

    _enabled = True
except ImportError:
    log.info(
        "pyopencl is not installed, ContextPyopencl will not be available"
    )
    cl = ModuleNotAvailable(
        message=(
            "pyopencl is not installed. ContextPyopencl is not available!"
        )
    )
    cl.Buffer = cl
    cla = cl
    _enabled = False

from ._patch_pyopencl_array import _patch_pyopencl_array

openclheader: List[SourceType] = [
    """\
#ifndef XOBJ_STDINT
typedef long           int64_t;
typedef int            int32_t;
typedef short          int16_t;
typedef char           int8_t;
typedef unsigned long  uint64_t;
typedef unsigned int   uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char  uint8_t;
#endif
#ifndef NULL
#define NULL 0L
#endif
"""
]

if _enabled:
    # order of base classes matters as it defines which __setitem__ is used
    class LinkedArrayPyopencl(BaseLinkedArray, cla.Array):
        @classmethod
        def _build_view(cls, a):
            assert len(a.shape) == 1
            return cls(
                cq=a.queue,
                shape=a.shape,
                dtype=a.dtype,
                data=a.base_data,
                offset=a.offset,
                strides=a.strides,
                order="C",
                _flags=a.flags,
            )


class ContextPyopencl(XContext):
    @property
    def nplike_array_type(self):
        return cla.Array

    @property
    def linked_array_type(self):
        return LinkedArrayPyopencl

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
            print(f"Platform {ip}  : {platform.name}")
            for id, device in enumerate(platform.get_devices()):
                print(f"Device   {ip}.{id}: {device.name}")

    def __init__(
        self, device=None, patch_pyopencl_array=True, minimum_alignment=None
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

        # TODO assume one device only
        if device is None:
            self.context = cl.create_some_context(interactive=False)
            self.device = self.context.devices[0]
            self.platform = self.device.platform
        else:
            if isinstance(device, str):
                platform, device = map(int, device.split("."))
                self.platform = cl.get_platforms()[platform]
                self.device = self.platform.get_devices()[device]
            else:
                self.device = device
                self.platform = device.platform

            self.context = cl.Context([self.device])

        self.queue = cl.CommandQueue(self.context)

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
        while i < 2**16:
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

    def build_kernels(
        self,
        sources,
        kernel_descriptions,
        specialize=True,
        apply_to_source=(),
        save_source_as=None,
        extra_cdef=None,
        extra_classes=(),
        extra_headers=(),
        compile=True,  # noqa
    ) -> Dict[Tuple[str, tuple], "KernelPyopencl"]:
        if not compile:
            raise NotImplementedError("compile=False available only on CPU.")

        classes = list(classes_from_kernels(kernel_descriptions))
        classes += list(extra_classes)
        classes = sort_classes(classes)

        # Update the kernel descriptions with the overriden classes
        cls_for_name = {cls.__name__: cls for cls in classes}
        for kernel_name, kernel in kernel_descriptions.items():
            for arg in kernel.args:
                arg.atype = cls_for_name.get(arg.atype.__name__, arg.atype)

        cls_sources = sources_from_classes(classes)

        headers = openclheader + list(extra_headers)

        sources = headers + cls_sources + sources

        source, folders = _concatenate_sources(sources, apply_to_source)

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

        out_kernels = {}
        for pyname, kernel in kernel_descriptions.items():
            if kernel.c_name is None:
                kernel.c_name = pyname

            out_kernels[pyname] = KernelPyopencl(
                function=getattr(prg, kernel.c_name),
                description=kernel,
                context=self,
            )

            out_kernels[pyname].source = source
            out_kernels[pyname].specialized_source = specialized_source

        return out_kernels

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

    @property
    def splike_lib(self):
        """
        Scipy features are not available through openCL
        """
        raise NotImplementedError

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
            src_offset=src_offset,
            dst_offset=dest_offset,
            byte_count=byte_count,
        )

    def write(self, offset, data):
        # From python object with buffer interface on cpu
        # log.debug(f"write {self} {offset} {data}")
        cl.enqueue_copy(
            self.context.queue, self.buffer, data, src_offset=offset
        )

    def read(self, offset, size):
        # To bytearray on cpu
        data = bytearray(size)
        cl.enqueue_copy(
            self.context.queue, data, self.buffer, src_offset=offset
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
            dst_offset=offset,
            byte_count=nbytes,
        )

    def to_native(self, offset: int, nbytes: int):
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
            dst_offset=offset,
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

    def to_nparray(self, offset, dtype, shape):
        return self.to_nplike(offset, dtype, shape).get()

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
            src_offset=offset,
        )
        return data

    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel

        Can fail if offset is not a multiple of self.alignment

        """
        return self.buffer[offset : offset + nbytes]


class KernelPyopencl(object):
    def __init__(
        self,
        function,
        description,
        context,
        wait_on_call=True,
    ):
        self.function = function
        self.description = description
        self.context = context
        self.wait_on_call = wait_on_call

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
                assert (
                    value._buffer.context is self.context
                ), f"Incompatible context for argument `{arg.name}`"
                return value._buffer.buffer[value._offset :]
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

        if isinstance(self.description.n_threads, str):
            n_threads = kwargs[self.description.n_threads]
        else:
            n_threads = self.description.n_threads

        event = self.function(
            self.context.queue, (n_threads,), None, *arg_list
        )

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

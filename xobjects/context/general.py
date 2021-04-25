from typing import NamedTuple, Optional
from abc import ABC, abstractmethod
import logging
import weakref


"""

TODO:
    - add test get_a_buffer
    - Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
    - Consider Buffer[offset] to create View and avoid _offset in type API
"""


log = logging.getLogger(__name__)


def _align(offset, alignment):
    "round to nearest multiple of 8"
    return (offset + alignment - 1) & (-alignment)


class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)


class ModuleNotAvailable(object):
    def __init__(self, message="Module not available"):
        self.message = message

    def __getattr__(self, attr):
        raise NameError(self.message)


class XContext(ABC):
    def __init__(self):
        self._kernels = MinimalDotDict()
        self._buffers = []
        self.kernels_v2 = MinimalDotDict()

    def new_buffer(self, capacity=1048576):
        buf = self._make_buffer(capacity=capacity)
        self.buffers.append(weakref.finalize(buf, log.debug, f"free buf"))
        return buf

    @property
    def buffers(self):
        return self._buffers

    @property
    def kernels(self):
        return self._kernels

    @abstractmethod
    def _make_buffer(self, capacity):
        "return buffer"

    @abstractmethod
    def add_kernels(self, src_code="", src_files=[], kernel_descriptions={}):
        pass

    # @abstractmethod
    def add_kernels_v2(
        self,
        sources: list,
        kernels: dict,
        specialize: bool,
        save_source_as: str,
    ):
        pass

    @abstractmethod
    def nparray_to_context_array(self, arr):
        return arr

    @abstractmethod
    def nparray_from_context_array(self, dev_arr):
        return dev_arr

    @property
    @abstractmethod
    def nplike_lib(self):
        "return lib"

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def zeros(self, *args, **kwargs):
        "return arr"

    @abstractmethod
    def plan_FFT(self, data, axes):
        "return fft"


class XBuffer(ABC):
    def __init__(self, capacity=1048576, context=None):

        if context is None:
            self.context = self._make_context()
        else:
            self.context = context
        self.buffer = self._new_buffer(capacity)
        self.capacity = capacity
        self.chunks = [Chunk(0, capacity)]

    @abstractmethod
    def _make_context(self):
        "return a default context"

    def allocate(self, size, alignment=1):
        # find available free slot
        # and update free slot if exists
        sizepa = size + alignment - 1
        for chunk in self.chunks:
            if sizepa <= chunk.size:
                offset = chunk.start
                chunk.start += sizepa
                if chunk.size == 0:
                    self.chunks.remove(chunk)
                return _align(offset, alignment)

        # no free slot check if can be allocated then try to grow
        if sizepa > self.capacity:
            self.grow(sizepa)
        else:
            self.grow(self.capacity)

        # try again
        return self.allocate(sizepa)

    def grow(self, capacity):
        oldcapacity = self.capacity
        newcapacity = self.capacity + capacity
        newbuff = self._new_buffer(newcapacity)
        self.copy_to(newbuff)
        self.buffer = newbuff
        if self.chunks[-1].end == self.capacity:  # last chunk is at the end
            self.chunks[-1].end = newcapacity
        else:
            self.chunks.append(Chunk(oldcapacity, newcapacity))
        self.capacity = newcapacity

    def free(self, offset, size):
        nch = Chunk(offset, offset + size)
        # insert sorted
        if offset > self.chunks[-1].start:  # new chuck at the end
            self.chunks.append(nch)
        else:  # new chuck needs to be inserted
            for ic, ch in enumerate(self.chunks):
                if offset <= ch.start:
                    self.chunks.insert(ic, nch)
                    break
        # merge chunks
        pch = self.chunks[0]
        newchunks = [pch]
        for ch in self.chunks[1:]:
            if pch.overlaps(ch):
                pch.merge(ch)
            else:
                newchunks.append(ch)
                pch = ch
        self.chunks = newchunks

    @abstractmethod
    def _new_buffer(self, capacity):
        "return newbuffer"

    @abstractmethod
    def update_from_native(self, offset, source, source_offset, nbytes):
        """Copy data from native buffer into self.buffer starting from offset"""

    @abstractmethod
    def copy_native(self, offset, nbytes):
        """return native data with content at from offset and nbytes"""

    @abstractmethod
    def update_from_buffer(self, offset, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""

    @abstractmethod
    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""

    @abstractmethod
    def update_from_nplike(self, offset, dest_dtype, value):
        """update data from nplike matching dest_dtype"""

    @abstractmethod
    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""

    @abstractmethod
    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel"""

    def update_from_xbuffer(self, offset, source, source_offset, nbytes):
        """update from any xbuffer, don't pass through gpu if possible"""
        if source.context == self.context:
            self.update_from_native(
                offset, source.buffer, source_offset, nbytes
            )
        else:
            data = source.to_bytearray(source_offset, nbytes)
            self.update_from_buffer(offset, data)

    def get_free(self):
        return sum([ch.size for ch in self.chunks])

    def __repr__(self):
        name = self.__class__.__name__
        return f"<{name} {self.get_free()}/{self.capacity}>"

    ##Old API
    @abstractmethod
    def copy_to(self, dest):
        pass

    @abstractmethod
    def copy_from(self, source, src_offset, dest_offset, byte_count):
        pass

    @abstractmethod
    def write(self, offset, data):
        pass

    @abstractmethod
    def read(self, offset, size):
        "return data"


class Chunk:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def size(self):
        return self.end - self.start

    #    def overlaps(self,other):
    #        return not ((other.end < self.start) or (other.start > self.end))

    def overlaps(self, other):
        return (other.end >= self.start) and (other.start <= self.end)

    def merge(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        return self

    def copy(self):
        return Chunk(self.start, self.end)

    def __repr__(self):
        return f"Chunk({self.start},{self.end})"


class View(NamedTuple):
    buffer: XBuffer
    offset: int
    size: int


available = []


class Arg:
    def __init__(
        self, atype, pointer=False, name=None, const=False, factory=None
    ):
        self.atype = atype
        self.pointer = pointer
        self.name = name
        self.const = const
        self.factory = factory

    def get_c_type(self):
        ctype = self.atype._c_type
        if self.pointer:
            ctype += "*"
        return ctype


class Kernel:
    def __init__(self, args, c_name=None, ret=None, n_threads=None):
        self.c_name = c_name
        self.args = args
        self.ret = ret
        self.n_threads = None


class Method:
    def __init__(self, kernel_name, arg_name="self"):
        self.kernel_name = kernel_name
        self.arg_name = arg_name

    def mk_method(self):
        def a_method(instance, *args, **kwargs):
            context = instance._buffer.context
            kernel = context.kernels[self.kernel_name]
            kwargs[self.arg_name] = instance
            return kernel(*args, **kwargs)

        return a_method

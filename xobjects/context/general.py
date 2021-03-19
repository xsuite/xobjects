from typing import NamedTuple
from abc import ABC, abstractmethod


"""

TODO:
    - add test get_a_buffer
    - Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
    - Consider Buffer[offset] to create View and avoid _offset in type API
"""

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class ModuleNotAvailable(object):
    def __init__(self, message='Module not available'):
        self.message=message

    def __getattr__(self, attr):
        raise NameError(self.message)

class Context(ABC):

    def __init__(self):
        self._kernels = MinimalDotDict()
        self._buffers = []

    @property
    def buffers(self):
        return self._buffers

    @property
    def kernels(self):
        return self._kernels

    @abstractmethod
    def new_buffer(self, capacity):
        pass

    @abstractmethod
    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):
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
        return lib

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def zeros(self, *args, **kwargs):
        return arr

    @abstractmethod
    def plan_FFT(self, data, axes):
        return fft 


class Buffer(ABC):
    def __init__(self, capacity=1048576, context=None):

        if context is None:
            self.context = self._DefaultContext()
        else:
            self.context = context
        self.buffer = self._new_buffer(capacity)
        self.capacity = capacity
        self.chunks = [Chunk(0, capacity)]

    def allocate(self, size):
        # find available free slot
        # and update free slot if exists
        for chunk in self.chunks:
            if size <= chunk.size:
                offset = chunk.start
                chunk.start += size
                if chunk.size == 0:
                    self.chunks.remove(chunk)
                return offset

        # no free slot check if can be allocated then try to grow
        if size > self.capacity:
            self.grow(size)
        else:
            self.grow(self.capacity)

        # try again
        return self.allocate(size)

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
        return newbuffer

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
        return data


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
    context: None
    buffer: Buffer
    offset: int
    size: int

    @classmethod
    def _from_size(cls, size, context=None, buffer=None, offset=None):
        if buffer is None:
            if offset is not None:
                raise ValueError("Cannot set `offset` without buffer")
            if context is None:
                context = ContextDefault()
            buffer = context.new_buffer(size)
        if offset is None:
            offset = buffer.allocate(size)
        return cls(context, buffer, offset, size)



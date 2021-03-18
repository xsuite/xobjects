from typing import NamedTuple


"""

TODO:
    - add test get_a_buffer
    - Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
    - Consider Buffer[offset] to create View and avoid _offset in type API
"""



class Buffer:
    def __init__(self, capacity=1048576, context=None):
        if context is None:
            from . import ContextDefault
            self.context = ContextDefault()
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

def get_a_buffer(size, context=None, buffer=None, offset=None):
    if buffer is None:
        if offset is not None:
            raise ValueError("Cannot set `offset` without buffer")
        if context is None:
            from . import ContextDefault
            context = ContextDefault()
        buffer = context.new_buffer(size)
    if offset is None:
        offset = buffer.allocate(size)
    return buffer, offset


class View(NamedTuple):
    context: None
    buffer: Buffer
    offset: int
    size: int

    @classmethod
    def _from_sise(cls, size, context=None, buffer=None, offset=None):
        if buffer is None:
            if offset is not None:
                raise ValueError("Cannot set `offset` without buffer")
            if context is None:
                context = ContextDefault()
            buffer = context.new_buffer(size)
        if offset is None:
            offset = buffer.allocate(size)
        return cls(context, buffer, offset, size)


def dispatch_arg(f, arg):
    if isinstance(arg, tuple):
        return f(*arg)
    elif isinstance(arg, dict):
        return f(**arg)
    else:
        return f(arg)


class Info:
    def __init__(self, **nargs):
        self.__dict__.update(nargs)

    def __repr__(self):
        args = [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
        return f"Info({','.join(args)})"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

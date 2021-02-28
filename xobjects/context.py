import weakref
import pyopencl


class CLContext:
    @classmethod
    def print_devices(cls):
        for ip, platform in enumerate(pyopencl.get_platforms()):
            print(f"Platform {ip}: {platform.name}")
            for id, device in enumerate(platform.get_devices()):
                print(f"Device {ip}.{id}: {device.name}")

    def __init__(self, device="0.0"):
        if isinstance(device, str):
            platform, device = map(int, device.split("."))
        else:
            self.device = device
            self.platform = device.platform

        self.platform = pyopencl.get_platforms()[platform]
        self.device = self.platform.get_devices()[device]
        self.context = pyopencl.Context([self.device])
        self.queue = pyopencl.CommandQueue(self.context)
        self.buffers = []

    def new_buffer(self, capacity=1048576):
        buf = CLBuffer(capacity=capacity, context=self)
        self.buffers.append(weakref.finalize(buf, print, "free", repr(buf)))
        return buf


class ByteArrayContext:
    def __init__(self):
        self.buffers = []

    def new_buffer(self, capacity=1048576):
        buf = ByteArrayBuffer(capacity=capacity, context=self)
        self.buffers.append(weakref.finalize(buf, print, "free", repr(buf)))
        return buf


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


class Buffer:
    def __init__(self, capacity=1048576, context=None):
        if context is None:
            self.context = ByteArrayContext()
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


class ByteArrayBuffer(Buffer):
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


class CLBuffer(Buffer):
    def _new_buffer(self, capacity):
        return pyopencl.Buffer(
            self.context.context, pyopencl.mem_flags.READ_WRITE, capacity
        )

    def copy_to(self, dest):
        pyopencl.enqueue_copy(self.context.queue, dest, self.buffer)

    def copy_from(self, source, src_offset, dest_offset, byte_count):
        pyopencl.enqueue_copy(
            self.context.queue, self.buffer, source, src_offset, dest_offset, byte_count
        )

    def write(self, offset, data):
        pyopencl.enqueue_copy(
            self.context.queue, self.buffer, data, device_offset=offset
        )

    def read(self, offset, size):
        data = bytearray(size)
        pyopencl.enqueue_copy(
            self.context.queue, data, self.buffer, device_offset=offset
        )
        return data

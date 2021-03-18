import weakref

from .general import Buffer

class ByteArrayContext:
    def __init__(self):
        self.buffers = []

    def new_buffer(self, capacity=1048576):
        buf = ByteArrayBuffer(capacity=capacity, context=self)
        self.buffers.append(weakref.finalize(buf, print, "free", repr(buf)))
        return buf

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

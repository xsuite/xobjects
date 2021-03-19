import weakref

import pyopencl

from .general import Buffer

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

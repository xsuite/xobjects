# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo


class Check:
    def __init__(self, ctx, capacity):
        self.ctx = ctx
        self.buffer = ctx.new_buffer(capacity=capacity)
        self.state = {}

    def random_string(self, maxlength=100):
        size = np.random.randint(1, maxlength)
        self.new_string(size)

    def new_string(self, size=100):
        if size > 0:
            data = bytes(np.random.randint(65, 90, size, dtype="u1"))

            offset = self.buffer.allocate(len(data))
            self.buffer.update_from_buffer(offset, data)
            self.state[offset] = data
            return offset
        else:
            raise ValueError("size must be >0")

    def free_string(self, offset):
        size = len(self.state[offset])
        self.buffer.free(offset, size)
        del self.state[offset]

    def random_free(self):
        ii = np.random.randint(1, len(self.state))
        offset = list(self.state.keys())[ii]
        self.free_string(offset)

    def check(self):
        for offset, value in self.state.items():
            assert self.buffer.to_bytearray(offset, len(value)) == value


def test_cl_print_devices():
    xo.ContextPyopencl.print_devices()


def test_cl_init():
    ctx = xo.ContextPyopencl(device="0.0")


def test_new_buffer():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        buff1 = ctx.new_buffer()
        buff2 = ctx.new_buffer(capacity=200)


def test_read_write():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        buff = ctx.new_buffer()
        bb = b"asdfasdfafsdf"
        buff.update_from_buffer(23, bb)
        assert buff.to_bytearray(23, len(bb)) == bb


def test_to_from_byterarray():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        buff = ctx.new_buffer()
        bb = b"asdfasdfafsdf"
        buff.update_from_buffer(23, bb)
        assert buff.to_bytearray(23, len(bb)) == bb


def test_allocate_simple():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        ch = Check(ctx, 200)
        ch.new_string(30)
        ch.check()


def test_free_simple():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        ch = Check(ctx, 200)
        offsets = [ch.new_string(ii * 2 + 1) for ii in range(10)]
        print(offsets)
        for offset in offsets:
            print(offset)
            ch.free_string(offset)
            ch.check()


def test_grow():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        ch = Check(ctx, 200)
        st = ch.new_string(150)
        st = ch.new_string(60)
        ch.check()
        assert ch.buffer.capacity == 400
        assert ch.buffer.chunks[0].start == st + 60
        assert ch.buffer.chunks[0].end == 400
        st = ch.new_string(500)
        assert ch.buffer.capacity == 900 + ch.buffer.default_alignment - 1
        ch.check()


def test_random_string():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        ch = Check(ctx, 200)

        for i in range(50):
            ch.random_string(maxlength=2000)
            ch.check()

        for i in range(50):
            ch.random_string(maxlength=2000)
            ch.check()
            ch.random_free()
            ch.check()


def test_nplike():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")
        buff = ctx.new_buffer(capacity=80)
        arr = np.arange(6.0).reshape((2, 3))
        offset = 3
        buff.update_from_buffer(offset, arr.tobytes())
        arr2 = buff.to_nplike(offset, "float64", (2, 3))
        arr3 = ctx.nparray_from_context_array(arr2)
        assert np.all(arr == arr3)


def test_type_matrix():
    try:
        import pyopencl
        import cupy
    except ImportError:
        return

    import numpy as np

    sources = []
    sources.append(bytearray(24))
    sources.append(np.zeros(24, dtype="uint8"))
    sources.append(np.zeros((6, 4), dtype="double"))
    sources.append(np.zeros((6, 2, 4), dtype="double")[:, 1, :])

    try:
        import pyopencl
        import pyopencl.array

        ctx = pyopencl.create_some_context(0)
        queue = pyopencl.CommandQueue(ctx)
        sources.append(pyopencl.Buffer(ctx, pyopencl.mem_flags.READ_WRITE, 24))
        sources.append(pyopencl.array.Array(queue, shape=24, dtype="uint8"))
        sources.append(
            pyopencl.array.Array(queue, shape=(6, 2, 4), dtype="uint8")
        )[:, 1, :]
    except:
        pass

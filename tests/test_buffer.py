# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import pytest

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts, requires_context


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


@requires_context("ContextPyopencl")
def test_cl_print_devices():
    xo.ContextPyopencl.print_devices()


@requires_context("ContextPyopencl")
def test_cl_init():
    _ = xo.ContextPyopencl(device="0.0")


@for_all_test_contexts
def test_new_buffer(test_context):
    _ = test_context.new_buffer()
    _ = test_context.new_buffer(capacity=200)


@for_all_test_contexts
def test_read_write(test_context):
    buff = test_context.new_buffer()
    bb = b"asdfasdfafsdf"
    buff.update_from_buffer(23, bb)
    assert buff.to_bytearray(23, len(bb)) == bb


@for_all_test_contexts
def test_to_from_byterarray(test_context):
    buff = test_context.new_buffer()
    bb = b"asdfasdfafsdf"
    buff.update_from_buffer(23, bb)
    assert buff.to_bytearray(23, len(bb)) == bb


@for_all_test_contexts
def test_allocate_simple(test_context):
    ch = Check(test_context, 200)
    ch.new_string(30)
    ch.check()


@for_all_test_contexts
def test_free_simple(test_context):
    ch = Check(test_context, 200)
    offsets = [ch.new_string(ii * 2 + 1) for ii in range(10)]
    print(offsets)
    for offset in offsets:
        print(offset)
        ch.free_string(offset)
        ch.check()

@for_all_test_contexts
def test_free(test_context):
    class CheckFree(xo.Struct):
        a = xo.Float64
    ch = CheckFree(a=5, _context=test_context)
    assert ch._buffer.capacity == 8
    assert ch._buffer.chunks == []
    with pytest.raises(ValueError, match="Cannot free outside of buffer"):
        ch._buffer.free(-2, 8)
    with pytest.raises(ValueError, match="Cannot free outside of buffer"):
        ch._buffer.free(0, 10)
    with pytest.raises(ValueError, match="Cannot free outside of buffer"):
        ch._buffer.free(7,2)
    ch._buffer.free(0,4)
    assert len(ch._buffer.chunks) == 1
    assert ch._buffer.chunks[0].start == 0
    assert ch._buffer.chunks[0].end == 4
    ch._buffer.free(0,4)  # Does nothing
    ch._buffer.free(2,4)  # Increases free chunk
    assert len(ch._buffer.chunks) == 1
    assert ch._buffer.chunks[0].start == 0
    assert ch._buffer.chunks[0].end == 6
    ch._buffer.free(7,1)
    assert len(ch._buffer.chunks) == 2
    assert ch._buffer.chunks[0].start == 0
    assert ch._buffer.chunks[0].end == 6
    assert ch._buffer.chunks[1].start == 7
    assert ch._buffer.chunks[1].end == 8


@for_all_test_contexts
def test_grow(test_context):
    ch = Check(test_context, 200)
    st = ch.new_string(150)
    st = ch.new_string(60)
    ch.check()
    assert ch.buffer.capacity == 400
    assert ch.buffer.chunks[0].start == st + 60
    assert ch.buffer.chunks[0].end == 400
    st = ch.new_string(500)
    assert ch.buffer.capacity == 900 + ch.buffer.default_alignment - 1
    ch.check()


@for_all_test_contexts
def test_random_string(test_context):
    ch = Check(test_context, 200)

    for i in range(50):
        ch.random_string(maxlength=2000)
        ch.check()

    for i in range(50):
        ch.random_string(maxlength=2000)
        ch.check()
        ch.random_free()
        ch.check()


@for_all_test_contexts
def test_nplike(test_context):
    buff = test_context.new_buffer(capacity=80)
    arr = np.arange(6.0).reshape((2, 3))
    offset = 3
    buff.update_from_buffer(offset, arr.tobytes())
    arr2 = buff.to_nplike(offset, "float64", (2, 3))
    arr3 = test_context.nparray_from_context_array(arr2)
    assert np.all(arr == arr3)

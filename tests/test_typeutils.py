# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
from xobjects.typeutils import allocate_on_buffer

import pytest


def test_allocate_on_buffer_ctx_none_buff_none_offset_none():
    buffer, offset = allocate_on_buffer(size=16)
    assert buffer.context.__class__ == xo.context_default.__class__
    assert offset == 0
    assert buffer.capacity == 16


def test_allocate_on_buffer_ctx_none_buff_none_offset_given():
    with pytest.raises(ValueError):
        _ = allocate_on_buffer(size=16, offset=5)


def test_allocate_on_buffer_ctx_none_buff_given_offset_none():
    original_buffer = xo.context_default.new_buffer(16)
    buffer, offset = allocate_on_buffer(size=16, buffer=original_buffer)
    assert buffer is original_buffer
    assert buffer.capacity >= 16


def test_allocate_on_buffer_ctx_none_buff_given_offset_given():
    original_buffer = xo.context_default.new_buffer(16)
    buffer, offset = allocate_on_buffer(
        size=16, buffer=original_buffer, offset=3
    )
    assert buffer is original_buffer
    assert offset == 3


def test_allocate_on_buffer_ctx_given_buff_none_offset_none():
    context = xo.ContextCpu()
    buffer, offset = allocate_on_buffer(size=32, context=context)
    assert buffer.context is context
    assert offset == 0
    assert buffer.capacity >= 32


def test_allocate_on_buffer_ctx_given_buff_none_offset_given():
    with pytest.raises(ValueError):
        _ = allocate_on_buffer(size=16, offset=5)


def test_allocate_on_buffer_ctx_given_buff_given_offset_none_ok():
    context = xo.ContextCpu()
    original_buffer = context.new_buffer(16)
    buffer, offset = allocate_on_buffer(
        size=32, context=context, buffer=original_buffer
    )
    assert buffer is original_buffer
    assert buffer.context is context
    assert offset == 0
    assert buffer.capacity >= 32


def test_allocate_on_buffer_ctx_given_buff_given_offset_none():
    context1, context2 = xo.ContextCpu(), xo.ContextCpu()
    assert context1 is not context2

    buffer_on_context1 = context1.new_buffer(16)

    with pytest.raises(ValueError):
        _ = allocate_on_buffer(
            size=16,
            context=context2,
            buffer=buffer_on_context1,
        )


@pytest.mark.parametrize("input_offset", ["packed", "aligned"])
def test_allocate_on_buffer_ctx_given_buff_given_offset_given_ok(input_offset):
    context = xo.ContextCpu()
    original_buffer = context.new_buffer(16)
    buffer, offset = allocate_on_buffer(
        size=32,
        context=context,
        buffer=original_buffer,
        offset=input_offset,
    )
    assert buffer is original_buffer
    assert buffer.context is context
    assert buffer.capacity >= 32


def test_allocate_on_buffer_ctx_given_buff_given_offset_given():
    context = xo.ContextCpu()
    original_buffer = context.new_buffer(10000)
    buffer, offset = allocate_on_buffer(
        size=32,
        context=context,
        buffer=original_buffer,
        offset=1234,
    )
    assert buffer is original_buffer
    assert offset == 1234

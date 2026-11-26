# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import ctypes
import numpy as np
import pytest

from xobjects.scalar import Complex64, Complex128
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_create_aligned_objects(test_context):
    for Arr in xo.Float64[3], xo.Float64[16], xo.Float64[17]:
        buff = test_context.new_buffer(10)
        assert buff.default_alignment == test_context.minimum_alignment
        for i in range(4):
            aa = Arr(_buffer=buff, _offset="aligned")
            print(test_context, Arr._size, aa._offset)
            assert aa._offset % test_context.minimum_alignment == 0


@pytest.mark.parametrize(
    "scalar, alignment",
    [
        (xo.UInt8, 1),
        (xo.Int16, 2),
        (xo.Int32, 4),
        (xo.Float64, 8),
        (Complex64, 4),
        (Complex128, 8),
    ],
)
def test_type_alignment(scalar, alignment):
    assert scalar._alignment == alignment
    assert scalar[13]._alignment == scalar._alignment
    assert scalar[13]._size == 13 * scalar._size
    assert scalar[:]._alignment == max(8, scalar._alignment)


def test_struct_matches_native_layout():
    class Native(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_uint8),
            ("b", ctypes.c_uint8),
            ("c", ctypes.c_int16),
            ("d", ctypes.c_float),
            ("e", ctypes.c_double),
            ("f", ctypes.c_uint8 * 13),
            ("g", ctypes.c_uint8),
        ]

    class Compact(xo.Struct):
        a = xo.UInt8
        b = xo.UInt8
        c = xo.Int16
        d = xo.Float32
        e = xo.Float64
        f = xo.UInt8[13]
        g = xo.UInt8

    assert Compact._alignment == ctypes.alignment(Native)
    assert Compact._size == ctypes.sizeof(Native)
    for field in Compact._fields:
        assert field.offset == getattr(Native, field.name).offset

    values = dict(a=1, b=2, c=3, d=4, e=5, f=list(range(13)), g=6)
    obj = Compact(**values)
    native = Native.from_buffer_copy(
        obj._buffer.to_bytearray(obj._offset, obj._size)
    )
    assert list(native.f) == values["f"]
    assert native.g == 6


@for_all_test_contexts
def test_nested_complex_alignment(test_context):
    class Inner(xo.Struct):
        flag = xo.UInt8
        small = Complex64
        large = Complex128
        tail = xo.UInt8

    class Outer(xo.Struct):
        flag = xo.UInt8
        items = Inner[2]
        tail = xo.UInt8

    assert [ff.offset for ff in Inner._fields] == [0, 4, 16, 32]
    assert Inner._alignment == 8
    assert Inner._size == 40
    assert Outer.items.offset == 8
    assert Outer._alignment == 8
    assert Outer._size == 96
    values = dict(flag=1, small=2 + 3j, large=4 + 5j, tail=6)
    obj = Outer(items=[values, values], _context=test_context)
    for item in obj.items:
        assert item._offset % 8 == 0
        assert item.small == values["small"]
        assert item.large == values["large"]
        assert item.tail == 6


@pytest.mark.parametrize(
    "shape, data_offset", [((None,), 16), ((2, None), 32), ((None, None), 40)]
)
@for_all_test_contexts
def test_dynamic_complex_array_alignment(shape, data_offset, test_context):
    array_type = Complex128[shape]
    values = np.arange(6).reshape((6,) if len(shape) == 1 else (2, 3))
    values = values.astype("complex128") + 2j
    array = array_type(values, _context=test_context)
    assert array_type._data_offset == data_offset
    np.testing.assert_array_equal(array.to_nparray(), values)
    restored = array_type._from_buffer(array._buffer, array._offset)
    np.testing.assert_array_equal(restored.to_nparray(), values)


@for_all_test_contexts
def test_dynamic_struct_alignment(test_context):
    class Dynamic(xo.Struct):
        tag = xo.UInt8
        tail = xo.UInt8
        text = xo.String
        values = Complex128[:]

    assert Dynamic._alignment == 8
    assert Dynamic.tag.offset == 8
    assert Dynamic.tail.offset == 9
    assert Dynamic.values.offset == 16  # Int64 offset slot
    assert Dynamic.text.offset == 24
    values = dict(tag=1, tail=2, text="x", values=[3 + 4j])
    obj = Dynamic(**values, _context=test_context)
    assert obj.values._offset - obj._offset == 40
    assert obj._size == 72
    assert obj.text == "x"
    assert obj.values[0] == 3 + 4j
    for array_type in (Dynamic[2], Dynamic[3], Dynamic[:]):
        count = array_type._shape[0] or 3
        array = array_type([values] * count, _context=test_context)
        restored = array_type._from_buffer(array._buffer, array._offset)
        for item in restored:
            assert item._offset % 8 == 0
            assert item.values._offset % 8 == 0
            assert item.text == "x"
            assert item.values[0] == 3 + 4j


@pytest.mark.parametrize("default_alignment", [1, 8, 128, 512])
@pytest.mark.parametrize("offset", [None, "aligned"])
def test_allocation_honors_type_and_buffer_alignment(
    default_alignment, offset
):
    buffer = xo.ContextCpu().new_buffer(1)
    buffer.default_alignment = default_alignment
    xo.UInt8[1](_buffer=buffer)
    obj = Complex128[1]([1 + 2j], _buffer=buffer, _offset=offset)
    assert obj._offset == max(8, default_alignment)
    assert obj[0] == 1 + 2j


@pytest.mark.parametrize("offset", ["packed", 1])
def test_explicit_offsets_bypass_allocation_alignment(offset):
    buffer = xo.ContextCpu().new_buffer(128)
    buffer.default_alignment = 128
    xo.UInt8[1](_buffer=buffer)
    obj = Complex128[1]([1 + 2j], _buffer=buffer, _offset=offset)
    assert obj._offset == 1
    assert obj[0] == 1 + 2j


@for_all_test_contexts
def test_first_dynamic_field_alignment(test_context):
    class FirstDynamic(xo.Struct):
        tag = xo.UInt8
        values = Complex128[:]

    assert FirstDynamic.values.offset == 16
    obj = FirstDynamic(tag=1, values=[2 + 3j], _context=test_context)
    assert obj.values._offset % 8 == 0
    assert obj.values[0] == 2 + 3j


def test_fixed_string_array_stride():
    string_type = xo.String.fixed(10)
    for array_type in (string_type[3], string_type[:]):
        array = array_type([0, 0, 0])
        assert array._strides == (16,)
        for index in range(3):
            assert array._get_offset(index) % 8 == 0


def test_reference_and_union_alignment():
    class ByteStruct(xo.Struct):
        value = xo.UInt8

    class WideUnion(xo.RawUnion):
        byte = xo.UInt8
        complex = Complex128

    assert xo.Ref[ByteStruct]._alignment == 8
    assert xo.UnionRef._alignment == 8
    assert xo.String._alignment == 8
    assert WideUnion._alignment == 8
    assert WideUnion._size == 16


@for_all_test_contexts
def test_compact_struct_generated_accessors(test_context):
    class CompactKernel(xo.Struct):
        a = xo.UInt8
        b = xo.Int16
        bytes = xo.UInt8[3]
        tail = xo.UInt8

    # The generated C API accessors are device functions, not kernels (and
    # they return values), so they cannot be called directly on GPU contexts.
    # Instead, call them from a kernel and check what it reads and writes.
    src = """
    #include "xobjects/headers/common.h"

    GPUKERN void compact_accessors(CompactKernel obj, GPUGLMEM int64_t* out) {
        VECTORIZE_OVER(tid, 1)
            out[0] = CompactKernel_get_a(obj);
            out[1] = CompactKernel_get_b(obj);
            out[2] = CompactKernel_get_bytes(obj, 0);
            out[3] = CompactKernel_get_bytes(obj, 1);
            out[4] = CompactKernel_get_bytes(obj, 2);
            out[5] = CompactKernel_get_tail(obj);
            out[6] = *CompactKernel_getp_b(obj);
            out[7] = CompactKernel_getp1_bytes(obj, 0)[2];
            CompactKernel_set_tail(obj, 7);
        END_VECTORIZE
    }
    """
    test_context.add_kernels(
        sources=[src],
        kernels={
            "compact_accessors": xo.Kernel(
                args=[
                    xo.Arg(CompactKernel, name="obj"),
                    xo.Arg(xo.Int64, pointer=True, name="out"),
                ],
                n_threads=1,
            )
        },
    )

    obj = CompactKernel(
        a=1, b=-2, bytes=[3, 4, 5], tail=6, _context=test_context
    )
    assert CompactKernel._size == 8

    out = test_context.zeros(8, dtype=np.int64)
    test_context.kernels.compact_accessors(obj=obj, out=out)
    out = test_context.nparray_from_context_array(out)

    assert list(out) == [1, -2, 3, 4, 5, 6, -2, 5]
    assert obj.tail == 7
    assert obj.a == 1 and obj.b == -2 and list(obj.bytes) == [3, 4, 5]

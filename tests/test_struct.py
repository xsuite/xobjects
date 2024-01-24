# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import cffi
import pytest

import xobjects as xo
from xobjects.methods import method_from_source
from xobjects.test_helpers import for_all_test_contexts, requires_context


def test_static_struct_def():
    class StructA(xo.Struct):
        a = xo.Float64
        b = xo.Int8
        c = xo.Int64

    assert StructA._size is not None

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2


@for_all_test_contexts
def test_static_struct(test_context):
    class StructA(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Int8
        c = xo.Field(xo.Int64)

    assert StructA.a.index == 0

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2

    s = StructA(_context=test_context)

    assert s._size is not None
    assert s.a == 3.5
    assert s.b == 0
    assert s.c == 0.0

    s.a = 5.2
    assert s.a == 5.2
    s.c = 7
    assert s.c == 7
    s.b = -4


@for_all_test_contexts
def test_nested_struct(test_context):
    class StructB(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.Int64, default=-4)
        c = xo.Int8

    class StructC(xo.Struct):
        a = xo.Field(xo.Float64, default=3.6)
        b = xo.Field(StructB)
        c = xo.Field(xo.Int8, default=-1)

    assert StructB._size is not None
    assert StructC._size is not None

    b = StructC(_context=test_context)

    assert b._size is not None
    assert b.a == 3.6
    assert b.b.a == 3.5
    assert b.b.c == 0


@for_all_test_contexts
def test_dynamic_struct(test_context):
    class StructD(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    assert StructD._size is None

    d = StructD(b="this is a test", _context=test_context)
    assert d._size is not None
    assert d.a == 3.5
    assert d.b == "this is a test"
    assert d.c == -1


@for_all_test_contexts
def test_dynamic_nested_struct(test_context):
    class StructE(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    info = StructE._inspect_args(b="this is a test")
    assert info.size == 56
    assert info.field_offsets == {1: 32}

    class StructF(xo.Struct):
        e = xo.Field(xo.Float64, default=3.5)
        f = xo.Field(xo.Float64, default=1.5)
        g = xo.Field(StructE)
        h = xo.Field(xo.Int8, default=-1)

    assert StructE._size is None
    assert StructF._size is None

    info = StructF._inspect_args(g={"b": "this is a test"})
    assert info.size == 96
    assert info.field_offsets == {2: 40}

    s = StructF(g={"b": "this is a test"}, _context=test_context)
    assert s._size is not None
    assert s.e == 3.5
    assert s.f == 1.5
    assert s.g.b == "this is a test"


@for_all_test_contexts
def test_assign_full_struct(test_context):
    class StructE(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    class StructF(xo.Struct):
        e = xo.Field(xo.Float64, default=3.5)
        f = xo.Field(xo.Float64, default=1.5)
        g = xo.Field(StructE)
        h = xo.Field(xo.Int8, default=-1)

    assert StructE._size is None
    assert StructF._size is None

    s = StructF(g={"b": "this is a test"}, _context=test_context)
    assert s._size is not None
    assert s.e == 3.5
    assert s.f == 1.5
    assert s.g.b == "this is a test"

    e = StructE(b="hello")
    s.g = e
    # assert f.h==-1


def test_init_from_xobj():
    class StructA(xo.Struct):
        a = xo.Float64
        b = xo.Float64

    s1 = StructA(a=1.3, b=2.4)
    s2 = StructA(s1)
    s3 = StructA(s1, _buffer=s1._buffer)

    assert s2.a == s1.a
    assert s3.b == s1.b


def test_nestednested():
    class MyStructA(xo.Struct):
        a = xo.Float64[:]
        b = xo.Float64[:]

    class MyStructB(xo.Struct):
        s = MyStructA

    b = MyStructB(s={"a": 10, "b": 10})

    assert b.s.a._size == 96
    assert b.s.b._size == 96
    assert b.s._size == 216
    assert b._size == 232


def test_copy_dynamic():
    class MyStruct(xo.Struct):
        a = xo.Float64
        b = xo.Float64[:]
        c = xo.Float64[:]

    s1 = MyStruct(a=2, b=[3, 4], c=[5, 6])
    s2 = MyStruct(s1)
    assert s1.a == s2.a
    assert s1.b[1] == s2.b[1]
    s1.b[1] = 33
    assert s2.b[1] == 4


def test_struct_allow_single_xo_inheritance():
    class Base(xo.Struct):
        f1 = xo.UInt16

    class Base2(xo.Struct):
        f2 = xo.UInt64

    with pytest.raises(TypeError) as e:
        class Child(Base, Base2):  # noqa
            f3 = xo.Float64

    assert 'multiple' in str(e.value).lower()


def test_struct_inheritance():
    class Base(xo.Struct):
        f1 = xo.UInt32
        f2 = xo.UInt64

    class Child(Base):
        f3 = xo.UInt16

    child = Child(f1=12, f2=43, f3=67)

    assert child.f1 == 12
    assert child.f2 == 43
    assert child.f3 == 67


def test_struct_method():
    class Base(xo.Struct):
        f1 = xo.UInt32[:]
        f2 = xo.UInt64

        method_sum = method_from_source(
            """
            uint64_t Base_method_sum(Base this, uint64_t factor) {
                uint64_t sum = Base_get_f2(this);
                for (int64_t ii = 0; ii < Base_len_f1(this); ii++) {
                    sum += Base_get_f1(this, ii);
                }
                return sum * factor;
            }
            """,
            name='Base_method_sum',
            return_type=xo.Arg(xo.UInt64),
            args=[
                xo.Arg(name='factor', atype=xo.UInt64),
            ],
        )

    instance = Base(f1=[1, 2, 3], f2=4)

    instance.compile_kernels()

    result = instance.method_sum(factor=2)
    assert result == 20


@requires_context("ContextCpu")
def test_compile_kernels_only_if_needed(tmp_path, mocker):
    """Test the use case of xtrack.

    We build a class with a kernel, verify that the kernel works, and then we
    save the kernel to a file. We then reload the kernel from the file and
    verify that it still works on a fresh context.
    """
    test_context = xo.ContextCpu()

    class TestClass(xo.HybridClass):
        _xofields = {
            "x": xo.Float64,
            "y": xo.Float64,
        }
        _extra_c_sources = [
            """
            /*gpufun*/ double myfun(TestClassData tc){
                double x = TestClassData_get_x(tc);
                double y = TestClassData_get_y(tc);
                return x * y;
            }
        """
        ]
        _kernels = {
            "myfun": xo.Kernel(
                args=[
                    xo.Arg(xo.ThisClass, name="tc"),
                ],
                ret=xo.Arg(xo.Float64),
            ),
        }

        def myfun(self):
            return self._context.kernels.myfun(tc=self)

    tc = TestClass(x=3, y=4, _context=test_context)
    tc.compile_kernels(only_if_needed=True)
    assert tc.myfun() == 12

    # Do it again, but this time we make sure that the kernel is not recompiled
    cffi_compile = mocker.patch.object(cffi.FFI, "compile")
    tc = TestClass(x=5, y=7, _context=test_context)
    tc.compile_kernels(only_if_needed=True)
    assert tc.myfun() == 35
    cffi_compile.assert_not_called()

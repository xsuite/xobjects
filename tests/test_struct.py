# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


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
    assert info.size == 48
    assert info._offsets == {1: 24}

    class StructF(xo.Struct):
        e = xo.Field(xo.Float64, default=3.5)
        f = xo.Field(xo.Float64, default=1.5)
        g = xo.Field(StructE)
        h = xo.Field(xo.Int8, default=-1)

    assert StructE._size is None
    assert StructF._size is None

    info = StructF._inspect_args(g={"b": "this is a test"})
    assert info.size == 80
    assert info._offsets == {2: 32}

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


def test_preinit():
    import numpy as np

    class Rotation(xo.Struct):
        cx = xo.Float64
        sx = xo.Float64

        @classmethod
        def _pre_init(cls, angle=0, **kwargs):
            rad = np.deg2rad(angle)
            kwargs["cx"] = np.cos(rad)
            kwargs["sx"] = np.sin(rad)
            return (), kwargs

        def _post_init(self):
            assert self.cx**2 + self.sx**2 == 1

        def myprint(self):
            return self.cx, self.sx

    rot = Rotation(angle=90)

    assert rot.sx == 1.0


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
    assert b.s._size == 208
    assert b._size == 216


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

import xobjects as xo

from xobjects.typeutils import Info
from xobjects.context import available


def test_static_struct_def():
    class StructA(xo.Struct):
        a = xo.Float64
        b = xo.Int8
        c = xo.Int64

    assert StructA._size is not None

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2


def test_static_struct():
    class StructA(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Int8
        c = xo.Field(xo.Int64)

    assert StructA.a.index == 0

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        s = StructA(_context=xo.ContextCpu())

        assert s._size is not None
        assert s.a == 3.5
        assert s.b == 0
        assert s.c == 0.0

        s.a = 5.2
        assert s.a == 5.2
        s.c = 7
        assert s.c == 7
        s.b = -4


def test_nested_struct():
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

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        b = StructC(_context=ctx)

        assert b._size is not None
        assert b.a == 3.6
        assert b.b.a == 3.5
        assert b.b.c == 0


def test_dynamic_struct():
    class StructD(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    assert StructD._size is None

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        d = StructD(b="this is a test", _context=ctx)
        assert d._size is not None
        assert d.a == 3.5
        assert d.b == "this is a test"
        assert d.c == -1


def test_dynamic_nested_struct():
    class StructE(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    info = StructE._inspect_args(b="this is a test")
    assert info == Info(size=48, _offsets={1: 24}, extra={})

    class StructF(xo.Struct):
        e = xo.Field(xo.Float64, default=3.5)
        f = xo.Field(xo.Float64, default=1.5)
        g = xo.Field(StructE)
        h = xo.Field(xo.Int8, default=-1)

    assert StructE._size is None
    assert StructF._size is None

    info = StructF._inspect_args(g={"b": "this is a test"})
    assert info == Info(
        size=80,
        _offsets={2: 32},
        extra={2: Info(size=48, _offsets={1: 24}, extra={})},
    )

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        s = StructF(g={"b": "this is a test"}, _context=ctx)
        assert s._size is not None
        assert s.e == 3.5
        assert s.f == 1.5
        assert s.g.b == "this is a test"


def test_assign_full_struct():
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

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        s = StructF(g={"b": "this is a test"}, _context=ctx)
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
        def _pre_init(cls, angle=0):
            rad = np.deg2rad(angle)
            return {"cx": np.cos(rad), "sx": np.sin(rad)}

        def _post_init(self):
            assert self.cx ** 2 + self.sx ** 2 == 1

        def myprint(self):
            return self.cx, self.sx

    rot = Rotation(angle=90)

    assert rot.sx == 1.0

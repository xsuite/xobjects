import xobjects as xo

from xobjects.typeutils import Info
from xobjects.context import available


def test_static_struct_def():
    class StructA(xo.Struct):
        a = xo.Float64
        b = xo.Int8
        c = xo.Int64

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2


def test_static_struct():
    #class StructA(xo.Struct):
    #    a = xo.Field(xo.Float64, default=3.5)
    #    b = xo.Field(xo.Int8, default=-4)
    #    c = xo.Field(xo.Int64, default=-1)
    class StructA(xo.Struct):
        a = xo.Float64
        b = xo.Int8
        c = xo.Int64

    assert StructA.a.index == 0
    assert StructA.b.index == 1
    assert StructA.c.index == 2

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f'Test {CTX}')
        ctx = CTX()

        s = StructA(_context=xo.ContextCpu())
        assert s.a == 3.5
        assert s.b == -4
        assert s.c == -1

        s.a = 5.2
        assert s.a == 5.2
        s.c = 7
        assert s.c == 7
        s.b = -4


def test_nested_struct():
    class StructB(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.Int64, default=-4)
        c = xo.Field(xo.Int8, default=-1)

    class StructC(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(StructB)
        c = xo.Field(xo.Int8, default=-1)

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f'Test {CTX}')
        ctx = CTX()

        b = StructC(_context=ctx)
        assert b.a == 3.5
        assert b.b.a == 3.5


def test_dynamic_struct():
    class StructD(xo.Struct):
        a = xo.Field(xo.Float64, default=3.5)
        b = xo.Field(xo.String, default=10)
        c = xo.Field(xo.Int8, default=-1)

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f'Test {CTX}')
        ctx = CTX()

        d = StructD(b="this is a test", _context=ctx)
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

    info = StructF._inspect_args(g={"b": "this is a test"})
    assert info == Info(
        size=80,
        _offsets={2: 32},
        extra={2: Info(size=48, _offsets={1: 24}, extra={})},
    )

    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f'Test {CTX}')
        ctx = CTX()

        s = StructF(g={"b": "this is a test"}, _context=ctx)
        assert s.e == 3.5
        assert s.f == 1.5
        assert s.g.b == "this is a test"
    # assert f.h==-1

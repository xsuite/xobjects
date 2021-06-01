import xobjects as xo


def test_classdef():
    class StructA(xo.Struct):
        fa = xo.Float64

    class ArrayB(xo.Float64[5]):
        pass

    class RefA(xo.UnionRef):
        _reftypes = (StructA, ArrayB)

    for ii, tt in enumerate(RefA._reftypes):
        RefA._typeid_from_type(tt) == ii
        RefA._typeid_from_name(tt.__name__) == ii
        RefA._type_from_typeid(ii) == tt
        RefA._type_from_name(tt.__name__) == tt


def test_init():
    class StructA(xo.Struct):
        fa = xo.Float64

    class ArrayB(xo.Float64[5]):
        pass

    class RefA(xo.UnionRef):
        _reftypes = (StructA, ArrayB)

    aref = RefA()
    assert aref.get() == None

    aref = RefA(None)
    assert aref.get() == None

    val = StructA(fa=3)
    aref = RefA(val, _buffer=val._buffer)
    assert aref.get().fa == 3.0

    val = StructA(fa=3)
    aref = RefA(val)
    assert aref.get().fa == 3.0

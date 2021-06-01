import xobjects as xo


def test_classdef():
    class StructA(xo.Struct):
        fa = xo.Float64

    class ArrayB(xo.Float64[5]):
        pass

    class RefA(xo.UnionRef):
        _refypes = (StructA, ArrayB)

    assert RefA._typeid_from_type(StructA) == 1

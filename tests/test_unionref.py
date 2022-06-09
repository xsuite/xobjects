# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

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


def test_array():
    class StructA(xo.Struct):
        fa = xo.Float64

    class ArrayB(xo.Float64[5]):
        pass

    class RefA(xo.UnionRef):
        _reftypes = (StructA, ArrayB)

    ArrNRefA = RefA[:]

    assert ArrNRefA.__name__ == "ArrNRefA"

    arr = ArrNRefA(10)

    assert arr[0] == None

    s1 = StructA(fa=3, _buffer=arr._buffer)
    arr[1] = s1

    assert arr[1].fa == 3.0
    s1.fa = 4
    assert arr[1].fa == 4.0

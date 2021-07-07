# pylint:disable=E1101

import numpy as np
import xobjects as xo

from xobjects import capi
from xobjects.typeutils import default_conf


def gen_classes():
    class Struct1(xo.Struct):
        field1 = xo.Int64
        field2 = xo.Float64

    class Struct2(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Float64[:]

    class Struct3(xo.Struct):
        field1 = xo.Float64
        field2 = xo.Float32[:]
        field3 = xo.Float64[:]
        field3 = xo.String

    class Struct2r(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Ref[xo.Float64[:]]

    class Struct3r(xo.Struct):
        field1 = xo.Float64
        field2 = xo.Ref[xo.Float32[:]]
        field3 = xo.Ref[xo.Float64[:]]

    class Struct4(xo.Struct):
        field1 = xo.Float64
        field2 = xo.Float32[:]
        field3 = xo.Float64[:]
        field3 = xo.Int8[:]

    class Struct5(xo.Struct):
        field1 = Struct1
        field2 = Struct2
        field2r = Struct2r
        field3 = Struct3
        field3r = Struct3r
        field4 = Struct4

    class URef(xo.UnionRef):
        _membertypes = [Struct1, Struct2]

    Array1 = xo.Int64[2]
    Array2 = xo.Int64[2, 3]
    Array3 = xo.Int64[2, 3, 5]
    Array4 = xo.Int64[None]
    Array5 = xo.Int64[2, None]
    Array6 = xo.Int64[None, 2]
    Array7 = xo.Int64[None, 3, None]
    Array8 = xo.Int64[2, None, 5]
    Array9 = xo.Int8[7]

    return locals().copy()

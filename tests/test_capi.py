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
        _reftypes = [Struct1, Struct2]

    Array1 = xo.Int64[2]
    Array2 = xo.Int64[:]
    Array3 = xo.Int64[2, 3]
    Array4 = xo.Int64[2, :]
    Array5 = xo.Int64[:, 2]
    Array6 = xo.Int64[2, 3, 5]
    Array7 = xo.Int64[:, 3, :]
    Array8 = xo.Int64[2, :, 5]
    Array9 = xo.Int8[3]
    Array10 = xo.Int8[:]
    Array11 = Struct1[3]
    Array12 = Struct2[3]
    Array12r = Struct2[3]
    Array13 = Struct3[3]
    Array13r = Struct3[3]
    Array14 = Struct4[3]
    Array15 = Struct5[3]

    res=type('', (object,), {})()
    res.__dict__.update(locals())

    return res




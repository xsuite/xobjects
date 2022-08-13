# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

# pylint:disable=E1101

import numpy as np
import xobjects as xo
import cffi

ffi = cffi.FFI()


def gen_classes():
    class Struct1(xo.Struct):
        field1 = xo.Int64
        field2 = xo.Float64

    class Struct2(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Float64[:]

    class Struct2r(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Ref[xo.Float64[:]]

    class Struct3(xo.Struct):
        field1 = xo.Float64
        field2 = xo.Float32[:]
        field3 = xo.Float64[:]
        field3 = xo.String

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

    res = type("", (object,), {})()
    res.__dict__.update(locals())

    return res


def test_struct1():
    class Struct1(xo.Struct):
        field1 = xo.Int64
        field2 = xo.Float64

    kernels = Struct1._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    s1 = Struct1(field1=2, field2=3.0)

    ctx.kernels.Struct1_set_field1(obj=s1, value=7)
    ctx.kernels.Struct1_get_field1(obj=s1) == s1.field1

    ctx.kernels.Struct1_set_field2(obj=s1, value=7)
    ctx.kernels.Struct1_get_field2(obj=s1) == s1.field2

    ps = ctx.kernels.Struct1_getp(obj=s1)
    p1 = ctx.kernels.Struct1_getp_field1(obj=s1)
    p2 = ctx.kernels.Struct1_getp_field2(obj=s1)

    assert ffi.cast("uint64_t *", ps)[0] == s1.field1
    assert ffi.cast("double *", ps)[1] == s1.field2
    assert p1[0] == s1.field1
    assert p2[0] == s1.field2


def test_array1():
    Array1 = xo.Int64[2]

    kernels = Array1._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    ini = [2, 7]

    a1 = Array1(ini)

    assert ctx.kernels.Arr2Int64_len(obj=a1) == len(ini)
    for ii, vv in enumerate(ini):
        a1[ii] = ii * 3
        assert ctx.kernels.Arr2Int64_get(obj=a1, i0=ii) == ii * 3
        ctx.kernels.Arr2Int64_set(obj=a1, i0=ii, value=ii * 4)
        assert a1[ii] == ii * 4


def test_array2():
    Array2 = xo.Int64[:]

    kernels = Array2._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    ini = [2, 7, 3]

    a1 = Array2(ini)

    assert ctx.kernels.ArrNInt64_len(obj=a1) == len(ini)
    for ii, vv in enumerate(ini):
        a1[ii] = ii * 3
        assert ctx.kernels.ArrNInt64_get(obj=a1, i0=ii) == ii * 3
        ctx.kernels.ArrNInt64_set(obj=a1, i0=ii, value=ii * 4)
        assert a1[ii] == ii * 4


def test_struct2():
    class Struct2(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Float64[:]

    s1 = Struct2(field1=2, field2=5)

    kernels = Struct2._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct2_len_field2(obj=s1) == len(s1.field2)
    for ii in range(len(s1.field2)):
        s1.field2[ii] = ii * 3
        assert ctx.kernels.Struct2_get_field2(obj=s1, i0=ii) == ii * 3
        ctx.kernels.Struct2_set_field2(obj=s1, i0=ii, value=ii * 4)
        assert s1.field2[ii] == ii * 4


def test_struct2r():
    class Struct2r(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Ref[xo.Float64[:]]

    s1 = Struct2r(field1=2)

    s1.field2 = 5

    kernels = Struct2r._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct2r_len_field2(obj=s1) == len(s1.field2)
    for ii in range(len(s1.field2)):
        s1.field2[ii] = ii * 3
        assert ctx.kernels.Struct2r_get_field2(obj=s1, i0=ii) == ii * 3
        ctx.kernels.Struct2r_set_field2(obj=s1, i0=ii, value=ii * 4)
        assert s1.field2[ii] == ii * 4


def test_unionref():
    class Struct1(xo.Struct):
        field1 = xo.Int64
        field2 = xo.Float64

    class Struct2(xo.Struct):
        field1 = xo.Int32
        field2 = xo.Float64[:]

    class URef(xo.UnionRef):
        _reftypes = [Struct1, Struct2]

    ArrNURef = URef[:]

    arr = ArrNURef(3)

    arr[0] = Struct1(field1=3, field2=4)
    arr[1] = Struct2(field1=2, field2=[5, 7])
    arr[2] = None

    kernels = ArrNURef._gen_kernels()
    kernels.update(Struct1._gen_kernels())
    kernels.update(Struct2._gen_kernels())
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    ctx.kernels.ArrNURef_typeid(obj=arr, i0=0) == URef._typeid_from_type(
        type(arr[0])
    )
    ctx.kernels.ArrNURef_typeid(obj=arr, i0=1) == URef._typeid_from_type(
        type(arr[1])
    )
    ctx.kernels.ArrNURef_typeid(obj=arr, i0=2) == -1

    p1 = ctx.kernels.Struct1_getp(obj=arr[0])
    p2 = ctx.kernels.ArrNURef_member(obj=arr, i0=0)

    assert int(ffi.cast("size_t", p1)) == int(ffi.cast("size_t", p2))


def test_get_two_indices():
    class Point(xo.Struct):
        x = xo.Float64
        y = xo.Float64

    class Triangle(Point[3]):
        pass

    class Mesh(Triangle[:]):
        pass

    kernels = Mesh._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)
    m = Mesh(5)
    m[0][1].x = 3
    assert ctx.kernels.Mesh_get_x(obj=m, i0=0, i1=1) == 3


def test_dependencies():
    import xobjects as xo

    class A(xo.Struct):
        a=xo.Float64[:]
        _extra_c_sources=["//blah blah A"]

    class C(xo.Struct):
        c=xo.Float64[:]
        _extra_c_sources=[" //blah blah C"]

    class B(xo.Struct):
        b=A
        c=xo.Float64[:]
        _extra_c_sources=[" //blah blah B"]
        _depends_on=[C]

    assert xo.context.sort_classes([B])[1:]==[A,C,B]



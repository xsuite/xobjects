# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

# pylint:disable=E1101

import pytest
import math
import numpy as np
import cffi

import xobjects as xo


ffi = cffi.FFI()


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
    field4 = xo.String


class Struct3r(xo.Struct):
    field1 = xo.Float64
    field2 = xo.Ref[xo.Float32[:]]
    field3 = xo.Ref[xo.Float64[:]]


class Struct4(xo.Struct):
    field1 = xo.Float64
    field2 = xo.Float32[:]
    field3 = xo.Float64[:]
    field4 = xo.Int8[:]


class Struct5(xo.Struct):
    field1 = Struct1
    field2 = Struct2
    field2r = Struct2r
    field3 = Struct3
    field3r = Struct3r
    field4 = Struct4


class DynLenType(xo.Struct):
    field1 = xo.Int32[:]


class URef(xo.UnionRef):
    _reftypes = [Struct1, Struct2]


@pytest.mark.parametrize(
    "array_cls, example_shape",
    [
        (xo.Int64[2], (2,)),
        (xo.Int64[:], (2,)),
        (xo.Int8[3], (3,)),
        (xo.Int8[:], (3,)),
        (xo.Int32[4, 5], (4, 5)),
        (xo.Int32[4, :], (4, 5)),
        (xo.Int32[:, 5], (4, 5)),
        (xo.Int64[2, 3, 5], (2, 3, 5)),
        (xo.Int64[:, 3, :], (2, 3, 5)),
        (xo.Int64[5, :, 2], (5, 13, 2)),
    ],
)
def test_array_static_type_init_get_set(array_cls, example_shape):
    c_name = array_cls.__name__
    length = math.prod(example_shape)

    kernels = array_cls._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    ini = np.arange(length).reshape(example_shape) + 7  # let's not have zeros

    a2 = array_cls(ini)

    len_fun = getattr(ctx.kernels, f"{c_name}_len")
    get_fun = getattr(ctx.kernels, f"{c_name}_get")
    set_fun = getattr(ctx.kernels, f"{c_name}_set")

    assert len_fun(obj=a2) == length
    for ii, vv in np.ndenumerate(ini):
        idx_kwargs = {f"i{dim}": jj for dim, jj in enumerate(ii)}

        a2[ii] = vv * 3
        assert get_fun(obj=a2, **idx_kwargs) == vv * 3

        set_fun(obj=a2, value=vv * 4, **idx_kwargs)
        assert a2[ii] == vv * 4


@pytest.mark.parametrize(
    "array_cls, example_shape",
    [
        (DynLenType[3], (3,)),
        (DynLenType[:], (3,)),
        (DynLenType[3, 4], (3, 4)),
        (DynLenType[:, 4], (3, 4)),
        (DynLenType[:, :], (3, 4)),
        (DynLenType[2, 3, 4], (2, 3, 4)),
        (DynLenType[2, :, 4], (2, 3, 4)),
        (DynLenType[:, 3, :], (2, 3, 4)),
    ],
)
def test_array_dynamic_type_init_get_set(array_cls, example_shape):
    c_name = array_cls.__name__
    length = math.prod(example_shape)

    kernels = array_cls._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels, save_source_as=f"{c_name}.c")

    numbers = [np.arange(ii) + ii + 3 for ii in range(length)]
    dt_fields = [DynLenType(field1=nums) for nums in numbers]

    numbers = np.array(numbers, dtype=object).reshape(example_shape)
    dt_fields = np.array(dt_fields).reshape(example_shape)

    arr = array_cls(dt_fields)

    len_fun = getattr(ctx.kernels, f"{c_name}_len")
    len_fun_f = getattr(
        ctx.kernels, f"{c_name}_len{len(example_shape)}_field1"
    )
    get_fun_f = getattr(ctx.kernels, f"{c_name}_get_field1")
    set_fun_f = getattr(ctx.kernels, f"{c_name}_set_field1")

    assert len_fun(obj=arr) == len(arr)

    for ii, field in np.ndenumerate(numbers):
        idx_kwargs = {f"i{dim}": jj for dim, jj in enumerate(ii)}
        assert len_fun_f(obj=arr, **idx_kwargs) == len(field)
        for idx_in_field, vv in enumerate(field):
            idx_kwargs[f"i{len(ii)}"] = idx_in_field
            assert get_fun_f(obj=arr, **idx_kwargs) == vv
            set_fun_f(obj=arr, value=13 * vv, **idx_kwargs)
            assert arr[ii].field1[idx_in_field] == 13 * vv


def test_struct1():
    kernels = Struct1._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    s1 = Struct1(field1=2, field2=3.0)

    ctx.kernels.Struct1_set_field1(obj=s1, value=7)
    assert ctx.kernels.Struct1_get_field1(obj=s1) == s1.field1

    ctx.kernels.Struct1_set_field2(obj=s1, value=7)
    assert ctx.kernels.Struct1_get_field2(obj=s1) == s1.field2

    ps = ctx.kernels.Struct1_getp(obj=s1)
    p1 = ctx.kernels.Struct1_getp_field1(obj=s1)
    p2 = ctx.kernels.Struct1_getp_field2(obj=s1)

    assert ffi.cast("uint64_t *", ps)[0] == s1.field1
    assert ffi.cast("double *", ps)[1] == s1.field2
    assert p1[0] == s1.field1
    assert p2[0] == s1.field2


def test_struct2():
    s2 = Struct2(field1=2, field2=5)

    kernels = Struct2._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct2_len_field2(obj=s2) == len(s2.field2)
    for ii in range(len(s2.field2)):
        s2.field2[ii] = ii * 3
        assert ctx.kernels.Struct2_get_field2(obj=s2, i0=ii) == ii * 3
        ctx.kernels.Struct2_set_field2(obj=s2, i0=ii, value=ii * 4)
        assert s2.field2[ii] == ii * 4


def test_struct2r():
    s2 = Struct2r(field1=2)
    s2.field2 = 5

    kernels = Struct2r._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct2r_len_field2(obj=s2) == len(s2.field2)
    for ii in range(len(s2.field2)):
        s2.field2[ii] = ii * 3
        assert ctx.kernels.Struct2r_get_field2(obj=s2, i0=ii) == ii * 3
        ctx.kernels.Struct2r_set_field2(obj=s2, i0=ii, value=ii * 4)
        assert s2.field2[ii] == ii * 4


def test_struct3():
    s3 = Struct3(
        field1=3,
        field2=[1, 2, 3, 4],
        field3=[11, 12, 13, 14, 15, 16, 17],
        field4="hello",
    )

    kernels = Struct3._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct3_len_field2(obj=s3) == 4
    assert ctx.kernels.Struct3_len_field3(obj=s3) == 7

    for ii in range(4):
        assert ctx.kernels.Struct3_get_field2(obj=s3, i0=ii) == ii + 1
        ctx.kernels.Struct3_set_field2(obj=s3, i0=ii, value=ii * 4)
        assert s3.field2[ii] == ii * 4

    for ii in range(7):
        assert ctx.kernels.Struct3_get_field3(obj=s3, i0=ii) == 11 + ii
        ctx.kernels.Struct3_set_field3(obj=s3, i0=ii, value=ii)
        assert s3.field3[ii] == ii


def test_struct3r():
    s3 = Struct3r(
        field1=3,
        field2=[1, 2, 3, 4],
        field3=[11, 12, 13, 14, 15, 16, 17],
    )

    kernels = Struct3r._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct3r_len_field2(obj=s3) == 4
    assert ctx.kernels.Struct3r_len_field3(obj=s3) == 7

    for ii in range(4):
        assert ctx.kernels.Struct3r_get_field2(obj=s3, i0=ii) == ii + 1
        ctx.kernels.Struct3r_set_field2(obj=s3, i0=ii, value=ii * 4)
        assert s3.field2[ii] == ii * 4

    for ii in range(7):
        assert ctx.kernels.Struct3r_get_field3(obj=s3, i0=ii) == 11 + ii
        ctx.kernels.Struct3r_set_field3(obj=s3, i0=ii, value=ii)
        assert s3.field3[ii] == ii


def test_struct4():
    af32 = [2.1, 2.2]
    af64 = [3.1, 3.2, 3.3]
    ai8 = [-9, -3, 11, 18]

    s4 = Struct4(
        field1=1.0,
        field2=af32,
        field3=af64,
        field4=ai8,
    )

    kernels = Struct4._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.Struct4_len_field2(obj=s4) == len(af32)
    assert ctx.kernels.Struct4_len_field3(obj=s4) == len(af64)
    assert ctx.kernels.Struct4_len_field4(obj=s4) == len(ai8)

    for ii, expected in enumerate(af32):
        assert math.isclose(
            ctx.kernels.Struct4_get_field2(obj=s4, i0=ii),
            expected,
            rel_tol=1e-7,  # single precision floats
        )
        ctx.kernels.Struct4_set_field2(obj=s4, i0=ii, value=7 * expected)
        assert math.isclose(
            ctx.kernels.Struct4_get_field2(obj=s4, i0=ii),
            7 * expected,
            rel_tol=1e-7,  # single precision float
        )

    for ii, expected in enumerate(af64):
        assert math.isclose(
            ctx.kernels.Struct4_get_field3(obj=s4, i0=ii), expected
        )
        ctx.kernels.Struct4_set_field3(obj=s4, i0=ii, value=7 * expected)
        assert math.isclose(
            ctx.kernels.Struct4_get_field3(obj=s4, i0=ii), 7 * expected
        )

    for ii, expected in enumerate(ai8):
        assert ctx.kernels.Struct4_get_field4(obj=s4, i0=ii) == expected
        ctx.kernels.Struct4_set_field4(obj=s4, i0=ii, value=7 * expected)
        assert ctx.kernels.Struct4_get_field4(obj=s4, i0=ii) == 7 * expected


def test_struct5():
    ctx = xo.ContextCpu()
    buff = ctx.new_buffer(0)

    s1 = Struct1(field1=2, field2=3.0, _buffer=buff)
    s2 = Struct2(field1=2, field2=[2.0, 2.0], _buffer=buff)
    s2r = Struct2r(field1=2, field2=s2.field2, _buffer=buff)
    s3 = Struct3(
        field1=3,
        field2=[1, 2, 3, 4],
        field3=[11, 12, 13, 14, 15, 16, 17],
        field4="hello",
        _buffer=buff,
    )
    s3r = Struct3r(
        field1=3,
        field2=s3.field2,
        field3=s3.field3,
        _buffer=buff,
    )
    s4 = Struct4(
        field1=1.0,
        field2=[2.1, 2.2],
        field3=[3.1, 3.2, 3.3],
        field4=[-9, -3, 11, 18],
        _buffer=buff,
    )
    s5 = Struct5(
        field1=s1,
        field2=s2,
        field2r=s2r,
        field3=s3,
        field3r=s3r,
        field4=s4,
        _buffer=buff,
    )
    s5.field2r = s5.field2
    s5.field3r = s5.field3

    kernels = Struct5._gen_kernels()
    kernels.update(Struct2._gen_kernels())
    kernels.update(Struct3._gen_kernels())
    ctx.add_kernels(kernels=kernels)
    ks = ctx.kernels

    # verify rebound references
    assert ks.Struct5_getp_field2r(obj=s5) != ks.Struct5_getp_field2(
        obj=s5
    )  # copy
    assert ks.Struct5_getp_field3r(obj=s5) != ks.Struct5_getp_field3(
        obj=s5
    )  # copy
    assert ks.Struct5_getp_field2r_field2(obj=s5) == ks.Struct2_getp_field2(
        obj=s5.field2
    )
    assert ks.Struct5_getp_field3r_field2(obj=s5) == ks.Struct3_getp_field2(
        obj=s5.field3
    )
    assert ks.Struct5_getp1_field2r_field2(
        obj=s5, i0=2
    ) == ks.Struct2_getp1_field2(obj=s5.field2, i0=2)
    assert ks.Struct5_getp1_field3r_field2(
        obj=s5, i0=2
    ) == ks.Struct3_getp1_field2(obj=s5.field3, i0=2)

    # set some nested values
    assert ks.Struct5_get_field1_field1(obj=s5) == s5.field1.field1
    ks.Struct5_set_field1_field1(obj=s5, value=10)
    assert ks.Struct5_get_field1_field1(obj=s5) == 10

    # check that references also work
    assert math.isclose(ks.Struct5_get_field2_field2(obj=s5, i0=0), 2)
    assert math.isclose(ks.Struct5_get_field2r_field2(obj=s5, i0=0), 2)
    ks.Struct5_set_field2r_field2(obj=s5, value=4, i0=0)
    assert math.isclose(ks.Struct5_get_field2_field2(obj=s5, i0=0), 4)
    assert math.isclose(ks.Struct5_get_field2r_field2(obj=s5, i0=0), 4)

    assert ks.Struct5_len_field3r_field3(obj=s5) == 7
    assert ks.Struct5_len_field3r_field3(
        obj=s5
    ) == ks.Struct5_len_field3_field3(obj=s5)


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
    kernels.update(URef._gen_kernels())
    kernels.update(Struct1._gen_kernels())
    kernels.update(Struct2._gen_kernels())
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.ArrNURef_typeid(
        obj=arr, i0=0
    ) == URef._typeid_from_type(type(arr[0]))
    assert ctx.kernels.ArrNURef_typeid(
        obj=arr, i0=1
    ) == URef._typeid_from_type(type(arr[1]))
    assert ctx.kernels.ArrNURef_typeid(obj=arr, i0=2) == -1

    for ii in range(2):
        p1 = ctx.kernels.Struct1_getp(obj=arr[ii])
        p2 = ctx.kernels.ArrNURef_member(obj=arr, i0=ii)
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
        a = xo.Float64[:]
        _extra_c_sources = ["//blah blah A"]

    class C(xo.Struct):
        c = xo.Float64[:]
        _extra_c_sources = [" //blah blah C"]

    class B(xo.Struct):
        b = A
        c = xo.Float64[:]
        _extra_c_sources = [" //blah blah B"]
        _depends_on = [C]

    assert xo.context.sort_classes([B])[1:] == [A, C, B]


def test_getp1_dyn_length_static_type_array():
    ArrNUint8 = xo.UInt8[:]
    char_array = ArrNUint8([42, 43, 44])

    kernels = ArrNUint8._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.ArrNUint8_len(obj=char_array) == 3

    s1 = ctx.kernels.ArrNUint8_getp1(obj=char_array, i0=0)
    s2 = ctx.kernels.ArrNUint8_getp1(obj=char_array, i0=1)
    s3 = ctx.kernels.ArrNUint8_getp1(obj=char_array, i0=2)

    assert s1[0] == 42
    assert s2[0] == 43
    assert s3[0] == 44


def test_getp1_dyn_length_dyn_type_array():
    ArrNUint8 = xo.UInt8[:]
    ArrNArr = ArrNUint8[:]
    ary = ArrNArr([[42, 43], [44, 45, 56]])

    kernels = ArrNArr._gen_kernels()
    kernels.update(ArrNUint8._gen_kernels())
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels, save_source_as="test-int.c")

    assert ctx.kernels.ArrNArrNUint8_len(obj=ary) == 2

    for ii in range(2):
        expected = ctx.kernels.ArrNUint8_getp(obj=ary[ii])
        result = ctx.kernels.ArrNArrNUint8_getp1(obj=ary, i0=ii)
        assert expected == result


def test_getp1_dyn_length_dyn_type_string_array():
    ArrNString = xo.String[:]
    string_array = ArrNString(["a", "bcdefghi", "jkl"])

    kernels = ArrNString._gen_kernels()
    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels, save_source_as="test.c")

    assert ctx.kernels.ArrNString_len(obj=string_array) == 3

    s0 = ctx.kernels.ArrNString_getp1(obj=string_array, i0=0)
    s1 = ctx.kernels.ArrNString_getp1(obj=string_array, i0=1)
    s2 = ctx.kernels.ArrNString_getp1(obj=string_array, i0=2)

    # Each string is encoded in Pascal-style, where the string
    # is preceded with its length (u64). Strings are also null
    # terminated.

    for ii, ch in enumerate(b"a\x00"):
        assert ord(ffi.cast("char *", s0)[8 + ii]) == ch

    for ii, ch in enumerate(b"bcdefghi\x00"):
        assert ord(ffi.cast("char *", s1)[8 + ii]) == ch

    for ii, ch in enumerate(b"jkl\x00"):
        assert ord(ffi.cast("char *", s2)[8 + ii]) == ch


def test_gpu_api():
    for ctx in xo.context.get_test_contexts():
        src_code = """
        /*gpufun*/
        void myfun(double x, double y,
            double* z){
            z[0] = x * y;
            }

        /*gpukern*/
        void my_mul(const int n,
            /*gpuglmem*/ const double* x1,
            /*gpuglmem*/ const double* x2,
            /*gpuglmem*/       double* y) {
            int tid = 0 //vectorize_over tid n
            double z;
            myfun(x1[tid], x2[tid], &z);
            y[tid] = z;
            //end_vectorize
            }
        """

        kernel_descriptions = {
            "my_mul": xo.Kernel(
                args=[
                    xo.Arg(xo.Int32, name="n"),
                    xo.Arg(xo.Float64, pointer=True, const=True, name="x1"),
                    xo.Arg(xo.Float64, pointer=True, const=True, name="x2"),
                    xo.Arg(xo.Float64, pointer=True, const=False, name="y"),
                ],
                n_threads="n",
            ),
        }

        ctx.add_kernels(
            sources=[src_code],
            kernels=kernel_descriptions,
            # save_src_as=f'_test_{name}.c')
            save_source_as=None,
            compile=True,
            extra_classes=[xo.String[:]],
        )

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo


def test_ref_to_static_type():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        buff = ctx._make_buffer(capacity=1024)

        Float64_3 = xo.Float64[3]

        arr1 = Float64_3([1, 2, 3], _buffer=buff)
        arr2 = Float64_3([4, 5, 6], _buffer=buff)

        class MyStructRef(xo.Struct):
            a = xo.Ref[Float64_3]
            # a = xo.Field(xo.Ref(Float64_3)) # More explicit

        assert MyStructRef._size == 8

        mystructref = MyStructRef(a=arr2, _buffer=buff)

        assert mystructref._size == 8
        assert (
            mystructref.a._offset == arr2._offset
        )  # Points to the same object
        for ii in range(3):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        assert (
            mystructref.a._offset == arr1._offset
        )  # Points to the same object
        for ii in range(3):
            assert mystructref.a[ii] == arr1[ii]

        mystructref.a = [7, 8, 9]
        for ii in range(3):
            assert mystructref.a[ii] == [7, 8, 9][ii]


def test_ref_to_dynamic_type():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        buff = ctx._make_buffer(capacity=1024)

        arr1 = xo.Float64[:]([1, 2, 3], _buffer=buff)
        arr2 = xo.Float64[:]([4, 5, 6, 7], _buffer=buff)

        class MyStructRef(xo.Struct):
            a = xo.Ref[xo.Float64[:]]

        assert MyStructRef._size == 8

        mystructref = MyStructRef(a=arr2, _buffer=buff)
        assert (
            mystructref.a._offset == arr2._offset
        )  # Points to the same object
        assert mystructref._size == 8
        for ii in range(4):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        assert (
            mystructref.a._offset == arr1._offset
        )  # Points to the same object
        for ii in range(3):
            assert mystructref.a[ii] == arr1[ii]

        mystructref.a = [
            7,
            8,
        ]
        for ii in range(2):
            assert mystructref.a[ii] == [7, 8, 9][ii]


def test_ref_c_api():
    for context in xo.context.get_test_contexts():
        print(f"Test {context}")

        class MyStruct(xo.Struct):
            a = xo.Float64[:]

        class MyStruct2(xo.Struct):
            a = xo.Float64[:]
            sr = xo.Ref(MyStruct)

        ms = MyStruct(a=[1, 2, 3], _context=context)

        ms2 = MyStruct2(_buffer=ms._buffer, sr=ms, a=[0, 0, 0])

        src = r"""
        /*gpukern*/
        void cp_sra_to_a(MyStruct2 ms, int64_t n){

            for(int64_t ii=0; ii<n; ii++){ //vectorize_over ii n
                double const val = MyStruct2_get_sr_a(ms, ii);
                MyStruct2_set_a(ms, ii, val);
            }//end_vectorize

        }
        """

        context.add_kernels(
            sources=[src],
            kernels={
                "cp_sra_to_a": xo.Kernel(
                    args=[
                        xo.Arg(MyStruct2, name="ms"),
                        xo.Arg(xo.Int64, name="n"),
                    ],
                    n_threads="n",
                )
            },
        )

        context.kernels.cp_sra_to_a(ms=ms2, n=len(ms.a))

        for vv, ww in zip(ms2.a, ms2.sr.a):
            assert vv == ww


def no_test_unionref():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        arr = xo.Float64[:]([1, 2, 3], _context=ctx)
        buf = arr._buffer
        string = xo.String("Test", _buffer=buf)

        class MyStructRef(xo.Struct):
            a = xo.Ref(xo.Float64[:], xo.String)

        assert MyStructRef._size == 16

        mystructref = MyStructRef(_buffer=buf)

        mystructref.a = arr
        assert (
            mystructref.a._offset == arr._offset
        )  # Points to the same object
        assert mystructref.a[1] == 2

        mystructref.a = string
        assert mystructref.a == "Test"


def no_test_array_of_unionrefs():
    class MyStructA(xo.Struct):
        a = xo.Float64

    class MyStructB(xo.Struct):
        a = xo.Int32

    Element = xo.Ref[MyStructA, MyStructB]
    ArrOfUnionRefs = Element[:]

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        aoref = ArrOfUnionRefs(10, _context=ctx)

        for ii in range(10):
            if np.mod(ii, 2) == 0:
                # Even elements
                temp = MyStructA()
            else:
                # Odd elements
                temp = MyStructB(a=10, _buffer=aoref._buffer)
            aoref[ii] = temp

        for ii in range(10):
            aoref[ii].a = 10 * ii

        for ii in range(10):
            print(f"aoref[{ii}]=", aoref[ii])
            assert aoref[ii].a == 10 * ii


import xobjects as xo


def test_unionref():
    class Triangle(xo.Struct):
        b = xo.Float64
        h = xo.Float64

        _extra_c_sources = ["""
            /*gpufun*/
            double Triangle_compute_area(Triangle tr, double scale){
                double b = Triangle_get_b(tr);
                double h = Triangle_get_h(tr);
                return 0.5*b*h*scale;
            }
            """]

    class Square(xo.Struct):
        a = xo.Float64

        _extra_c_sources = ["""
            /*gpufun*/
            double Square_compute_area(Square sq, double scale){
                double a = Square_get_a(sq);
                return a*a*scale;
            }
            """]

    class Base(xo.UnionRef):
        _reftypes = (Triangle, Square)
        _methods = [
            xo.Method(
                c_name="compute_area",
                args=[xo.Arg(xo.Float64, name="scale")],
                ret=xo.Arg(xo.Float64),
            )
        ]

    class Prism(xo.Struct):
        base = Base
        height = xo.Float64
        volume = xo.Float64

        _extra_c_sources = ["""
            /*gpukern*/
            void Prism_compute_volume(Prism pr){
                Base base = Prism_getp_base(pr);
                double height = Prism_get_height(pr);
                double base_area = Base_compute_area(base, 3.);
                printf("base_area = %e", base_area);
                Prism_set_volume(pr, base_area*height);
            }
            """]

    for context in xo.context.get_test_contexts():
        print(f"Test {context}")

        context.add_kernels(
            kernels={
                "Prism_compute_volume": xo.Kernel(
                    args=[xo.Arg(Prism, name="prism")]
                )
            }
        )

        triangle = Triangle(b=2, h=3, _context=context)
        prism_triangle = Prism(base=triangle, height=5, _context=context)
        square = Square(a=2, _context=context)
        prism_square = Prism(base=square, height=10, _context=context)

        context.kernels.Prism_compute_volume(prism=prism_triangle)
        context.kernels.Prism_compute_volume(prism=prism_square)

        assert prism_triangle.volume == 45
        assert prism_square.volume == 120

        assert prism_triangle._has_refs

def test_has_refs():

    class StructWRef(xo.Struct):
        a = xo.Ref(xo.Float64[:])
    assert StructWRef._has_refs

    class StructNoRef(xo.Struct):
        a = xo.Float64[:]
    assert not StructNoRef._has_refs

    class NestedWRef(xo.Struct):
        s = StructWRef
    assert NestedWRef._has_refs

    class NestedNoRef(xo.Struct):
        s = StructNoRef
    assert not NestedNoRef._has_refs

    ArrNoRef = xo.Float64[:]
    assert not ArrNoRef._has_refs

    ArrWRef = xo.Ref(xo.Float64)[:]
    assert ArrWRef._has_refs

    class StructArrRef(xo.Struct):
        arr = ArrWRef
    assert StructArrRef._has_refs

    class StructArrNoRef(xo.Struct):
        arr = ArrNoRef
    assert not StructArrNoRef._has_refs

    ArrOfStructRef = NestedWRef[:]
    assert ArrOfStructRef._has_refs

    class MyUnion(xo.UnionRef):
        _ref = [xo.Float64, xo.Int32]
    assert MyUnion._has_refs


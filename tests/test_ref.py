import xobjects as xo
from xobjects.context import available

def test_ref_to_static_type():
    for CTX in xo.ContextCupy, xo.ContextPyopencl, xo.ContextCpu:
        if CTX not in available:
            continue

        context = CTX()
        print(context)
        buff = context._make_buffer(capacity=1024)

        Float64_3 = xo.Float64[3]

        arr1 = Float64_3([1,2,3], _buffer=buff)
        arr2 = Float64_3([4,5,6], _buffer=buff)

        class MyStructRef(xo.Struct):
            a = xo.Ref[Float64_3]
            #a = xo.Field(xo.Ref(Float64_3)) # More explicit

        assert MyStructRef._size == 8

        mystructref = MyStructRef(a=arr2, _buffer=buff)
        assert mystructref._size == 8
        for ii in range(3):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        for ii in range(3):
            assert mystructref.a[ii] == arr1[ii]

        mystructref.a = [7,8,9]
        for ii in range(3):
            assert mystructref.a[ii] == [7,8,9][ii]


def test_ref_to_dynamic_type():
    for CTX in xo.ContextCupy, xo.ContextPyopencl, xo.ContextCpu:
        if CTX not in available:
            continue

        context = CTX()
        print(context)
        buff = context._make_buffer(capacity=1024)

        arr1 = xo.Float64[:]([1,2,3], _buffer=buff)
        arr2 = xo.Float64[:]([4,5,6,7], _buffer=buff)

        class MyStructRef(xo.Struct):
            a = xo.Ref[xo.Float64[:]]

        assert MyStructRef._size == 8

        mystructref = MyStructRef(a=arr2, _buffer=buff)
        assert mystructref._size == 8
        for ii in range(4):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        for ii in range(3):
            assert mystructref.a[ii] == arr1[ii]

        mystructref.a = [7,8,]
        for ii in range(2):
            assert mystructref.a[ii] == [7,8,9][ii]

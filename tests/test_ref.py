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
        assert mystructref.a._offset == arr2._offset # Points to the same object
        for ii in range(3):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        assert mystructref.a._offset == arr1._offset # Points to the same object
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
        assert mystructref.a._offset == arr2._offset # Points to the same object
        assert mystructref._size == 8
        for ii in range(4):
            assert mystructref.a[ii] == arr2[ii]

        mystructref.a = arr1
        assert mystructref.a._offset == arr1._offset # Points to the same object
        for ii in range(3):
            assert mystructref.a[ii] == arr1[ii]

        mystructref.a = [7,8,]
        for ii in range(2):
            assert mystructref.a[ii] == [7,8,9][ii]

def test_unionref():
    arr = xo.Float64[:]([1,2,3])
    buf = arr._buffer
    string = xo.String('Test', _buffer=buf)

    class MyStructRef(xo.Struct):
        a = xo.Ref([xo.Float64[:], xo.String])

    assert MyStructRef._size == 16

    mystructref = MyStructRef(_buffer=buf)

    mystructref.a = arr
    assert mystructref.a._offset == arr._offset # Points to the same object
    assert mystructref.a[1] == 2

    mystructref.a = string
    assert mystructref.a == 'Test'


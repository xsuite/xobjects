import numpy as np

import xobjects as xo

from xobjects.typeutils import Info


def test_get_shape():
    from xobjects.array import get_shape_from_array

    assert get_shape_from_array(0) == ()
    assert get_shape_from_array([]) == (0,)
    assert get_shape_from_array([1, 2, 3]) == (3,)
    assert get_shape_from_array(range(3)) == (3,)
    assert get_shape_from_array([[], []]) == (2, 0)
    assert get_shape_from_array([[2], [2]]) == (2, 1)


def mk_classes():
    class ArrayA(xo.Array):
        _itemtype = xo.Float64
        _shape = (6, 6)
        _order = (0, 1)

    class ArrayB(xo.Array):
        _itemtype = xo.Float64
        _shape = (6, None)
        _order = (0, 1)

    class ArrayC(xo.Array):
        _itemtype = xo.Float64
        _shape = (None,)
        _order = "C"

    class ArrayD(xo.Array):
        _itemtype = ArrayA
        _shape = (None,)
        _order = "F"

    class ArrayE(xo.Array):
        _itemtype = ArrayB
        _shape = (None,)
        _order = "C"

    return ArrayA, ArrayB, ArrayC, ArrayD, ArrayE


def test_class_creation():

    ArrayA, ArrayB, ArrayC, ArrayD, ArrayE = mk_classes()

    assert ArrayA._is_static_shape == True
    assert ArrayE._is_static_type == False
    assert ArrayA._size == 36 * 8

    for arr in ArrayB, ArrayC, ArrayD, ArrayE:
        assert arr._is_static_shape == False
    for arr in ArrayB, ArrayC, ArrayD, ArrayE:
        assert arr._is_static_shape == False

    assert ArrayA._data_offset == 0
    assert ArrayB._data_offset == 32
    assert ArrayC._data_offset == 16
    assert ArrayD._data_offset == 16
    assert ArrayE._data_offset == 16

    assert ArrayA._strides == (48, 8)
    assert ArrayC._strides == (8,)
    assert ArrayD._strides == (ArrayA._size,)


def test_class_mk_array():
    ArrayA = xo.Float64[3, 6]
    assert ArrayA._shape == (3, 6)
    assert ArrayA._order == (0, 1)
    assert ArrayA.__name__ == "Float64_3by6"

    ArrayA = xo.Float64[None, 6]
    assert ArrayA._shape == (None, 6)
    assert ArrayA.__name__ == "Float64_Nby6"

    ArrayA = xo.String[3:1, 4:0, 5:2]
    assert ArrayA._shape == (3, 4, 5)
    assert ArrayA.__name__ == "String_3by4by5"

    class StructA(xo.Struct):
        a = xo.Float64

    ArrayA = StructA[10]

    class StructB(xo.Struct):
        a = xo.Float64

    ArrayA = StructA[10]

    assert ArrayA.__name__ == "StructA_10"


def test_inspect_args():
    import numpy as np

    arrays = mk_classes()

    for AnArray in arrays:
        if AnArray._size is not None:
            info = AnArray._inspect_args()
            assert info.value == None

    ArrayA, ArrayB, ArrayC, ArrayD, ArrayE = arrays

    info = ArrayA._inspect_args(np.zeros((6, 6)))

    assert info.size == 36 * 8


def test_array_allocation():
    MyArray = xo.Int64[10]
    ss = MyArray()


def test_array_sshape_stype():
    Array1D = xo.Int64[3]
    Array2D = xo.Int64[2, 3]
    Array3D = xo.Int64[2, 3, 4]

    for cls in Array1D, Array2D, Array3D:
        ss = cls()
        for ii in ss._iter_index():
            assert ss[0] == 0
            ss[ii] = sum(ii) if len(ss._shape) > 1 else ii
        for ii in ss._iter_index():
            assert ss[ii] == (sum(ii) if len(ss._shape) > 1 else ii)


def test_array_dshape_stype():
    Array1 = xo.Int64[:]
    Array2 = xo.Int64[:, 3, 4]
    Array3 = xo.Int64[2, :, 4]
    Array4 = xo.Int64[2, 3, :]

    for cls in Array1, Array2, Array3, Array4:
        ss = cls(3)
        for ii in ss._iter_index():
            assert ss[0] == 0
            ss[ii] = sum(ii) if len(ss._shape) > 1 else ii
        for ii in ss._iter_index():
            assert ss[ii] == (sum(ii) if len(ss._shape) > 1 else ii)

    arr = Array1(10)
    arr[3] = 42
    data = arr._buffer.to_nplike("int64", (12,), arr._offset)
    assert data[0] == 8 + 8 + 10 * 8
    assert data[1] == 10
    data[6] = 43
    assert arr[4] == 43

    arr = Array2(2)
    arr[1, 2, 3] = 42
    data = arr._buffer.to_nplike("int64", (29,), arr._offset)

    assert data[0] == 8 + 8 + 3 * 8 + 24 * 8


def test_array_sshape_dtype():
    Array1 = xo.Int64[:]
    Array2 = Array1[3]
    arr = Array2([2, 3, 4])
    arr[0][1] = 3
    assert arr[0][1] == 3

    arr1 = Array1(2)
    assert arr1._shape == arr[0]._shape


def test_array_dshape_dtype():
    Array1 = xo.Int64[:]
    Array2 = Array1[:]
    ss = Array2([2, 3, 4])
    ss[0][1] = 3
    assert ss[0][1] == 3

    ss0 = Array1(2)
    ss1 = Array1(3)
    ss2 = Array1(4)
    assert ss0._shape == ss[0]._shape


def test_array_in_struct():
    class Multipole(xo.Struct):
        order = xo.Int64
        length = xo.Float64
        hxl = xo.Float64
        hyl = xo.Float64
        bal = xo.Float64[:]

    m = Multipole(order=2, bal=np.array([1.0, 2.0, 3.0, 4.0]))

import xobjects as xo

from xobjects.context import Info

def test_get_shape():
    from xobjects.array import get_shape_from_array

    assert get_shape_from_array(0)==()
    assert get_shape_from_array([])==(0,)
    assert get_shape_from_array([1,2,3])==(3,)
    assert get_shape_from_array(range(3))==(3,)
    assert get_shape_from_array([[],[]])==(2,0)
    assert get_shape_from_array([[2],[2]])==(2,1)


def mk_classes():
    class ArrayA(xo.Array):
        _itemtype=xo.Float64
        _shape = (6,6)
        _order = (0,1)

    class ArrayB(xo.Array):
        _itemtype=xo.Float64
        _shape = (6,None)
        _order = (0,1)

    class ArrayC(xo.Array):
        _itemtype=xo.Float64
        _order = 'C'

    class ArrayD(xo.Array):
        _itemtype=ArrayA
        _order = 'C'

    class ArrayE(xo.Array):
        _itemtype=ArrayB
        _order = 'C'

    return ArrayA,ArrayB,ArrayC,ArrayD,ArrayE


def test_class_creation():

    ArrayA,ArrayB,ArrayC,ArrayD,ArrayE = mk_classes()

    assert ArrayA._is_static_shape == True
    assert ArrayE._is_static_type == False
    assert ArrayA._size==36*8

    for arr in ArrayB,ArrayC,ArrayD,ArrayE:
        assert arr._is_static_shape == False
    for arr in ArrayB,ArrayC,ArrayD,ArrayE:
        assert arr._is_static_shape == False


def test_inspect_args():
    import numpy as np

    ArrayA,ArrayB,ArrayC,ArrayD,ArrayE = mk_classes()

    info=ArrayA._inspect_args(np.zeros((6,6)))

    assert info==Info(size=36*8)




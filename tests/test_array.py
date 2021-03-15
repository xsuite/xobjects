import xobjects as xo

def test_array_class():
    class ArrayA(xo.Array):
        _itemtype=xo.Float64
        _shape = (6,6)


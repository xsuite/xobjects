import xobjects as xo


def test_string_class():
    info = xo.String._inspect_args("abcde")
    assert info.size == 5 + 1 + 2 + 8


def test_string_fixed():
    cls = xo.String.fixed(10)
    assert cls.__name__ == "String10"
    assert cls._size == 10


def test_string_init1():
    ss = xo.String(10)
    assert ss._size is not None
    assert ss._buffer.capacity == 18


def test_string_init2():
    ss = xo.String("test")
    assert ss._size is not None
    assert ss.to_str() == "test"


def test_string_init3():
    for ctx in xo.context.get_test_contexts():
        ss = xo.String("test", _context=ctx)
        assert xo.String._from_buffer(ss._buffer, ss._offset) == "test"

def test_string_array():
    import numpy as np
    StringArray=xo.String[:]
    data=np.array(["asd","as"])
    for ctx in xo.context.get_test_contexts():
        xobj=StringArray(data,_context=ctx)
        assert data[1]==xobj[1]



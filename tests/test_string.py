import xobjects as xo
import xobjects.context as xoc


def test_string_class():
    info = xo.String._inspect_args("abcde")
    assert info.size == 5 + 1 + 2 + 8


def test_string_init1():
    ss = xo.String(10)
    assert ss._buffer.capacity == 18


def test_string_init2():
    ss = xo.String("test")
    assert ss.to_str() == "test"


def test_string_init3():
    for cls in [xoc.ContextCpu, xoc.ContextPyopencl, xoc.ContextCupy]:

        if cls in xoc.available:
            ctx = cls()
        else:
            continue
        print(repr(cls))
        ss = xo.String("test", _context=ctx)
        assert xo.String._from_buffer(ss._buffer, ss._offset) == "test"

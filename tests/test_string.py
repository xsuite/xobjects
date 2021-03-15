import xobjects as xo


def test_string_class():
    size, off = xo.String._get_size_from_args("abcde")
    assert size==5+8
    assert off==None


def test_string_init1():
    ss=xo.String(10)
    assert ss._buffer.capacity==18

def test_string_init2():
    ss=xo.String("test")
    assert ss.to_str()=="test"

def test_string_init3():
    for ctx in xo.ByteArrayContext(), xo.CLContext():
       ctx=xo.ByteArrayContext()
       ss=xo.String("test",_context=ctx)
       assert xo.String._from_buffer(ss._buffer,ss._offset)=="test"


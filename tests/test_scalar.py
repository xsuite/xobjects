import xobjects as xo

scalars=(
    xo.Float64,
    xo.Float32,
    xo.Int64,
    xo.UInt64,
    xo.Int32,
    xo.UInt32,
    xo.Int16,
    xo.UInt16,
    xo.Int8,
    xo.UInt8
)


def test_scalar():
    nn=123
    for ctx in xo.ByteArrayContext(), xo.CLContext():
        buff=ctx.new_buffer()
        for sc in scalars:
            offset=buff.allocate(sc._size)
            sc._to_buffer(buff,offset,nn)
            vv=sc._from_buffer(buff, offset)
            assert nn==vv
            assert nn==sc(nn)



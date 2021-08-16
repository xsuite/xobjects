import xobjects as xo


def test_create_aligned_objects():

    for Arr in xo.Float64[3], xo.Float64[16], xo.Float64[17]:
        for ctx in xo.context.get_test_contexts():
            buff = ctx.new_buffer(10)
            assert buff.default_alignment == ctx.minimum_alignment
            for i in range(4):
                aa = Arr(_buffer=buff, _offset="aligned")
                print(ctx, Arr._size, aa._offset)
                assert aa._offset % ctx.minimum_alignment == 0

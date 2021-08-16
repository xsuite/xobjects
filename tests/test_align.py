import xobjects as xo


def test_create_aligned_objects():
    Arr=xo.Float64[3]

    for ctx in xo.context.get_test_contexts():
        buff=ctx.new_buffer(2**16)
        assert buff.default_alignment==ctx.minimum_alignment
        for i in range(4):
            aa=Arr(_buffer=buff,_offset='aligned')
            print(ctx,aa._offset)
            assert aa._offset%ctx.minimum_alignment==0







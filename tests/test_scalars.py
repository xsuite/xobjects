# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

scalars = (
    xo.Float64,
    xo.Float32,
    xo.Int64,
    xo.UInt64,
    xo.Int32,
    xo.UInt32,
    xo.Int16,
    xo.UInt16,
    xo.Int8,
    xo.UInt8,
)


def test_scalar_class():
    for sc in scalars:
        info1 = sc._inspect_args(1.1)
        info2 = sc._inspect_args(1)
        assert info1.size == sc._size
        assert info2.size == sc._size


@for_all_test_contexts
def test_scalar_buffer(test_context):
    nn = 123
    buff = test_context.new_buffer()
    for sc in scalars:
        offset = buff.allocate(sc._size)
        sc._to_buffer(buff, offset, nn)
        vv = sc._from_buffer(buff, offset)
        assert nn == vv
        assert nn == sc(nn)

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_create_aligned_objects(test_context):
    for Arr in xo.Float64[3], xo.Float64[16], xo.Float64[17]:
        buff = test_context.new_buffer(10)
        assert buff.default_alignment == test_context.minimum_alignment
        for i in range(4):
            aa = Arr(_buffer=buff, _offset="aligned")
            print(test_context, Arr._size, aa._offset)
            assert aa._offset % test_context.minimum_alignment == 0

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo


def test_dressed_struct():

    class Element(xo.HybridClass):
        _xofields = {
            'n': xo.Int32,
            'b': xo.Float64,
            'vv': xo.Float64[:],
        }
        def __init__(self, vv=None, **kwargs):
            if "_xobject" in kwargs.keys():
                self.xoinitialize(**kwargs)
            else:
                self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv, **kwargs)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        ele = Element([1, 2, 3], _context=context)
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 2

        new_vv = context.nparray_to_context_array(np.array([7, 8, 9]))
        ele.vv = new_vv
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 8

        ele.n = 5.0
        assert ele.n == ele._xobject.n == 5

        ele.b = 50
        assert ele.b == ele._xobject.b == 50.0

        dd = ele.to_dict()
        assert dd["vv"][1] == 8
        assert isinstance(dd["vv"], np.ndarray)


def test_explicit_buffer():

    class Element(xo.HybridClass):
        _xofields = {
            'n': xo.Int32,
            'b': xo.Float64,
            'vv': xo.Float64[:],
        }
        def __init__(self, vv=None, **kwargs):
            self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv, **kwargs)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        ele1 = Element([1, 2, 3], _context=context)
        ele2 = Element([7, 8, 9], _buffer=ele1._buffer)

        assert ele1.vv[1] == ele1._xobject.vv[1] == 2
        assert ele2.vv[1] == ele2._xobject.vv[1] == 8
        for ee in [ele1, ele2]:
            assert ee._buffer is ee._xobject._buffer
            assert ee._offset == ee._xobject._offset

        assert ele1._buffer is ele2._buffer
        assert ele1._offset != ele2._offset

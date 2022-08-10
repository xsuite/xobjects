# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo

def test_linked_arrays():
    class MyStruct(xo.HybridClass):
        _xofields = {
            '_a': xo.Float64[:],
            '_asq': xo.Float64[:]}

        def __init__(self, a, **kwargs):
            super().__init__(_a=len(a), _asq=len(a), **kwargs)
            self.a = a

        @property
        def a(self):
            LinkedArrayType = self._buffer.context.linked_array_type
            return LinkedArrayType.from_array(
                                            self._a,
                                            mode='setitem_from_container',
                                            container=self,
                                            container_setitem_name='_a_setitem')
        @a.setter
        def a(self, value):
            self.a[:] = value

        def _a_setitem(self, indx, value):
            self._a[indx] = value
            self._asq[indx] = self._a[indx]**2

        @property
        def asq(self):
            LinkedArrayType = self._buffer.context.linked_array_type
            return LinkedArrayType.from_array(
                                            self._asq,
                                            mode='readonly')

    for context in xo.context.get_test_contexts():

        ctx2np = context.nparray_from_context_array
        np2ctx = context.nparray_to_context_array

        m = MyStruct(a=np2ctx(np.array([1.,2.,3.])), _context=context)
        assert np.all(ctx2np(m.a) == np.array([1,2,3]))
        assert np.all(ctx2np(m.asq) == np.array([1,4,9]))

        m.a[1:] = m.a[:2] + 10
        assert np.all(ctx2np(m.a) == np.array([1, 11, 12]))
        assert np.all(ctx2np(m.asq) == np.array([1, 121, 144]))

        m.a = 5
        assert np.all(ctx2np(m.a) == 5)
        assert np.all(ctx2np(m.asq) == 25)

        b = m.a.copy()
        assert np.all(ctx2np(m.a) == ctx2np(b))

        m.a = np2ctx(np.array([4,-5,6]))
        b = m.a.copy() + 3
        assert np.all(ctx2np(m.a - b)  == -3)
        assert np.all(ctx2np(b - m.a)  == 3)
        assert np.all(ctx2np(m.a/(m.a*2))  == 0.5)
        assert np.all(ctx2np(-m.a)  == ctx2np(-10*m.a/10))

        m.a = m.a*2
        assert np.all(ctx2np(m.a) == np.array([8, -10, 12]))
        assert np.all(ctx2np(m.asq) == np.array([64, 100, 144]))

        m.a = np2ctx(np.array([4,-5,6]))
        m.a *= 2
        assert np.all(ctx2np(m.a) == np.array([8, -10, 12]))
        assert np.all(ctx2np(m.asq) == np.array([64, 100, 144]))

        m.a = np2ctx(np.array([1, 2, 3]))
        m.a = m.a**2 +1
        assert np.all(ctx2np(m.a) == np.array([2, 5, 10]))
        assert np.all(ctx2np(m.asq) == np.array([4, 25, 100]))

        if not(isinstance(context, xo.ContextPyopencl)): # masking not working
            m.a = np2ctx(np.array([4,-5,6]))
            m.a[m.a<0] += m.a[:1]
            assert np.all(ctx2np(m.a) == np.array([4, -1, 6]))
            assert np.all(ctx2np(m.asq) == np.array([16, 1, 36]))

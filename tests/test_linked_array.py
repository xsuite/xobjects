import numpy as np
import xobjects as xo

class MyStruct(xo.DressedStruct):
    _xofields = {
        '_a': xo.Float64[:],
        '_asq': xo.Float64[:]}

    def __init__(self, a):
        super().__init__(_a=len(a), _asq=len(a))
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


m = MyStruct(a=[1,2,3])
assert np.all(m.a == np.array([1,2,3]))
assert np.all(m.asq == np.array([1,4,9]))

m.a[1:] = m.a[:2] + 10
assert np.all(m.a == np.array([1, 11, 12]))
assert np.all(m.asq == np.array([1, 121, 144]))

m.a = 5
assert np.all(m.a == 5)
assert np.all(m.asq == 25)

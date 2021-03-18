import numpy as np

"""
TODO:
    - make scalars as class instead of instances
"""

from .context import Info

class NumpyScalar:
    def __init__(self, dtype):
        self._dtype = np.dtype(dtype)
        self._size = self._dtype.itemsize

    def _from_buffer(self, buffer, offset):
        data = buffer.read(offset, self._size)
        return np.frombuffer(data, dtype=self._dtype)[0]

    def _to_buffer(self, buffer, offset, value, info=None):
        data = self._dtype.type(value).tobytes()
        buffer.write(offset, data)

    def __call__(self, value=0):
        return self._dtype.type(value)

    def _inspect_args(self,arg):
        return Info(size=self._size)


Float128 = NumpyScalar("float128")
Float64 = NumpyScalar("float64")
Float32 = NumpyScalar("float32")
Int64 = NumpyScalar("int64")
UInt64 = NumpyScalar("uint64")
Int32 = NumpyScalar("int32")
UInt32 = NumpyScalar("uint32")
Int16 = NumpyScalar("int16")
UInt16 = NumpyScalar("uint16")
Int8 = NumpyScalar("int8")
UInt8 = NumpyScalar("uint8")
Complex64 = NumpyScalar("complex64")
Complex128 = NumpyScalar("complex128")
Complex256 = NumpyScalar("complex256")

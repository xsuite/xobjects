# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""
Scalars: Types olding numbers


NB: scalars cannot be classes as classes() needs to return a numpy scalar, however in union isintance could be used (try to subclass numpy scalar?)
"""

import numpy as np
import logging
from .typeutils import Info


log = logging.getLogger(__name__)


class NumpyScalar:
    def __init__(self, dtype, cname):
        self.__name__ = dtype.capitalize()
        self._dtype = np.dtype(dtype)
        self._size = self._dtype.itemsize
        self._c_type = cname

    def _from_buffer(self, buffer, offset=0):
        data = buffer.to_bytearray(offset, self._size)
        return np.frombuffer(data, dtype=self._dtype)[0]

    def _to_buffer(self, buffer, offset, value, info=None):
        data = self._dtype.type(value).tobytes()
        buffer.update_from_buffer(offset, data)

    def __call__(self, value=0):
        return self._dtype.type(value)

    def _inspect_args(self, arg):
        return Info(size=self._size)

    def __getitem__(self, shape):
        from .array import Array  # avoid circular import

        return Array.mk_arrayclass(self, shape)

    def __repr__(self):
        return self.__name__

    def _array_to_buffer(self, buffer, offset, value):
        return buffer.update_from_buffer(offset, value.tobytes())

    def _array_from_buffer(self, buffer, offset, count):
        return buffer.to_nplike(offset, self._dtype, (count,))

    def _gen_data_paths(self, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [self])
        return paths


Float64 = NumpyScalar("float64", "double")
Float32 = NumpyScalar("float32", "float")
Int64 = NumpyScalar("int64", "int64_t")
UInt64 = NumpyScalar("uint64", "uint64_t")
Int32 = NumpyScalar("int32", "int32_t")
UInt32 = NumpyScalar("uint32", "uint32_t")
Int16 = NumpyScalar("int16", "int16_t")
UInt16 = NumpyScalar("uint16", "uint16_t")
Int8 = NumpyScalar("int8", "int8_t")
UInt8 = NumpyScalar("uint8", "uint8_t")
Complex64 = NumpyScalar("complex64", "float[2]")
Complex128 = NumpyScalar("complex128", "double[2]")
# Complex256 = NumpyScalar("complex256", "double[4]")  # incompatible with M1 CPU
# Float128 = NumpyScalar("float128", "double[2]")      # incompatible with M1 CPU


def is_scalar(cls):
    return isinstance(cls, NumpyScalar)


class Void:
    _c_type = "void"
    _size = None

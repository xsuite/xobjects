# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
"""Types representing scalars.

The classes in this file are used to handle scalars, and introduce XoType
classes for these types. The instances of scalars themselves though will be
numpy types. See the comment in XoScalar.__new__ for more details.
"""
from abc import ABC, abstractmethod

import numpy as np
import logging
from xobjects.base_type import XoType, XoTypeMeta, XoInstanceInfo

log = logging.getLogger(__name__)


class Void(XoType, ABC):
    """Void type to be used as a placeholder, e.g., in kernel signatures."""
    _c_type = 'void'


class XoScalarMeta(XoTypeMeta):
    """The type of XoScalar class.

    Notes
    -----
    Currently this class does not do anything special, but is useful for
    `isinstance` checking of scalar types. See `is_scalar` below. We could
    accomplish something similar with issubclass, but then we need another check
    for isinstance(cls, type), as issubclass expects the second parameter to
    be a class.
    """


def is_scalar(cls):
    return isinstance(cls, XoScalarMeta)


class XoScalar(XoType, metaclass=XoScalarMeta):
    _size: int
    _dtype: np.dtype
    _c_type: str

    def __new__(cls, value=0):
        """Creating a new instance of XoScalar will produce numpy scalars.

        As such instances of XoScalar cannot exist, see __init__.
        """
        # Returning np.dtype instance here lets us avoid the hassle of
        # subclassing np.number. However, it could be beneficial to explore
        # whether this alternative is less hacky.
        return cls._dtype.type(value)

    @abstractmethod
    def __init__(self):
        """This class cannot be instantiated: see __new__."""

    @classmethod
    def make_new_type(cls, dtype: str, c_name: str) -> XoScalarMeta:
        """Create a XoScalar class for a scalar type.

        Arguments
        ---------
        dtype: str
            Numpy dtype string for the scalar type.
        c_name: str
            The string representing the type in C.

        Returns
        -------
        Type[XoScalar]:
            The new XoScalar type.
        """
        dtype_instance = np.dtype(dtype)

        class _NewType(cls):
            _size = dtype_instance.itemsize
            _dtype = dtype_instance
            _c_type = c_name

        _NewType.__name__ = dtype.capitalize()
        return _NewType

    @classmethod
    def _get_size(cls, _: 'XoScalar'):
        return cls._size

    @classmethod
    def _inspect_args(cls, *args, **kwargs) -> XoInstanceInfo:
        return XoInstanceInfo(size=cls._size)

    @classmethod
    def _from_buffer(cls, buffer, offset=0):
        data = buffer.to_bytearray(offset, cls._size)
        return np.frombuffer(data, dtype=cls._dtype)[0]

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        data = cls._dtype.type(value).tobytes()
        buffer.update_from_buffer(offset, data)

    @classmethod
    def _array_to_buffer(cls, buffer, offset, value):
        """Write an array `value` of scalars of this type to a buffer.

        Notes
        -----
        This is used by `xo.Array` instead of iterating over the values and
        calling `_to_buffer`, as the latter would be very slow.
        """
        return buffer.update_from_buffer(offset, value.tobytes())

    @classmethod
    def _array_from_buffer(cls, buffer, offset, count):
        """Write an array `value` of scalars of this type to a buffer.

        Notes
        -----
        This is used by `xo.Array` instead of iterating over the values and
        calling `_from_buffer`, as the latter would be very slow.
        """
        return buffer.to_nplike(offset, cls._dtype, (count,))

    @classmethod
    def _gen_data_paths(cls, base=None):
        if base is None:
            return [[cls]]
        return [base + [cls]]


Float64 = XoScalar.make_new_type("float64", "double")
Float32 = XoScalar.make_new_type("float32", "float")
Int64 = XoScalar.make_new_type("int64", "int64_t")
UInt64 = XoScalar.make_new_type("uint64", "uint64_t")
Int32 = XoScalar.make_new_type("int32", "int32_t")
UInt32 = XoScalar.make_new_type("uint32", "uint32_t")
Int16 = XoScalar.make_new_type("int16", "int16_t")
UInt16 = XoScalar.make_new_type("uint16", "uint16_t")
Int8 = XoScalar.make_new_type("int8", "int8_t")
UInt8 = XoScalar.make_new_type("uint8", "uint8_t")
Complex64 = XoScalar.make_new_type("complex64", "float[2]")
Complex128 = XoScalar.make_new_type("complex128", "double[2]")

# Incompatible with M1 CPU
# Complex256 = XoScalar.make_new_type("complex256", "double[4]")
# Float128 = XoScalar.make_new_type("float128", "double[2]")

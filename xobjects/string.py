# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""
String: holds a null terminated variable length string

init:
    String(10)  create an empty string with  10 byte capacity (including termination \0)
    String("a string") create and initialize string with the minimal capacity

Layout:
    [capacity]
    data
    [null] at least 1 byte


String class data:
    _size = None
    _c_type = "char*"

String instance data:
    _size = string type size (including capacity)


TODO:
- String._to_buffer to use same context copy
- consider caching  the length
- consider using __slots__
- consider adding size in the class
"""
from dataclasses import dataclass

from .base_type import XoTypeMeta, XoType, XoInstanceInfo
from .typeutils import allocate_on_buffer, _to_slot_size, is_integer, _is_dynamic

from .scalar import Int64
from .array import Array

import logging

log = logging.getLogger(__name__)


@dataclass
class StringInstanceInfo(XoInstanceInfo):
    str_content: bytes = None


class MetaString(XoTypeMeta):
    """The metaclass for the Xobjects string type."""


class String(XoType, metaclass=MetaString):
    """The Xobjects string type."""
    _c_type = "char*"

    def __init__(
        self, string_or_int, _buffer=None, _offset=None, _context=None
    ):
        info = self._inspect_args(string_or_int)
        size = info.size
        self._buffer, self._offset = allocate_on_buffer(
            size, _context, _buffer, _offset
        )

        self._to_buffer(
            self._buffer, self._offset, string_or_int, info=info
        )

        self._size = size

    @classmethod
    def _inspect_args(cls, string_or_int) -> StringInstanceInfo:
        if _is_dynamic(cls):
            if isinstance(string_or_int, int):
                return StringInstanceInfo(size=string_or_int + 8)
            elif isinstance(string_or_int, str):
                data = bytes(string_or_int, "utf8")
                size = _to_slot_size(len(data) + 1 + 8)
                return StringInstanceInfo(size=size, str_content=data)  # add zero termination
            elif isinstance(string_or_int, cls):
                return StringInstanceInfo(size=string_or_int._size)
            raise ValueError(
                f"String can accept only one integer or string and not `{string_or_int}`"
            )
        else:
            data = bytes(string_or_int, "utf8")
            return StringInstanceInfo(size=cls._size, str_content=data)

    @classmethod
    def _from_buffer(cls, buffer, offset=0, encoding="utf8"):
        # TODO keep in mind that in windows many functions return wchar encoded in utf16
        return cls._get_data(buffer, offset).decode(encoding).rstrip("\x00")

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info: StringInstanceInfo = None):
        if isinstance(value, String):
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, value._size
            )
            return

        if info is None:
            info = cls._inspect_args(value)

        size = info.size
        string_capacity = info.size

        if _is_dynamic(cls):
            Int64._to_buffer(buffer, offset, size)
            string_capacity -= 8
            offset += 8

        if isinstance(value, str):
            data = info.str_content
            off = string_capacity - len(data)
            data += b"\x00" * off
            buffer.update_from_buffer(offset, data)
        elif is_integer(value):
            pass
        else:
            raise ValueError(f"{value} not a string")

    @classmethod
    def _get_data(cls, buffer, offset):
        if _is_dynamic(cls):
            length = Int64._from_buffer(buffer, offset) - 8
            offset += 8
        else:
            length = cls._size
        return buffer.to_bytearray(offset, length)

    @classmethod
    def fixed(cls, size):
        """Create a static string class with capacity of `size` bytes."""
        if is_integer(size) and size > 0:
            name = f"String{size}"
            data = dict(_size=size)
            return type(cls)(name, (String,), data)
        else:
            raise ValueError("String needs a positive integer")

    @classmethod
    def _get_size(cls, instance: 'String'):
        if _is_dynamic(cls):
            return Int64._from_buffer(instance._buffer, instance._offset)
        return cls._size

    def to_str(self):
        return self._from_buffer(self._buffer, self._offset)

    def to_bytes(self):
        return self._get_data(self._buffer, self._offset)

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        return paths


def is_string(atype):
    return isinstance(atype, MetaString)

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

from .typeutils import allocate_on_buffer, Info, _to_slot_size, is_integer

from .scalar import Int64
from .array import Array

import logging

log = logging.getLogger(__name__)


class MetaString(type):
    def __getitem__(self, shape):
        return Array.mk_arrayclass(self, shape)

    def _inspect_args(cls, string_or_int):
        if cls._size is None:
            if isinstance(string_or_int, int):
                return Info(size=string_or_int + 8)
            elif isinstance(string_or_int, str):
                data = bytes(string_or_int, "utf8")
                size = _to_slot_size(len(data) + 1 + 8)
                return Info(size=size, data=data)  # add zero termination
            elif isinstance(string_or_int, cls):
                return Info(size=string_or_int._size)
            raise ValueError(
                f"String can accept only one integer or string and not `{string_or_int}`"
            )
        else:
            return Info(size=cls._size)

    def _to_buffer(cls, buffer, offset, value, info=None):
        # log.debug(f"{cls} to buffer {offset}  `{value}`")
        if info is None:
            info = cls._inspect_args(value)
        size = info.size
        string_capacity = info.size - 8
        Int64._to_buffer(buffer, offset, size)
        if isinstance(value, String):
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, value._size
            )
        elif isinstance(value, str):
            data = info.data
            off = string_capacity - len(data)
            data += b"\x00" * off
            # log.debug(f"to_buffer {offset+8} {len(data)} {string_capacity}")
            buffer.update_from_buffer(offset + 8, data)
        elif is_integer(value):
            pass
        else:
            raise ValueError(f"{value} not a string")

    def _get_data(cls, buffer, offset):
        ll = Int64._from_buffer(buffer, offset)
        return buffer.to_bytearray(offset + 8, ll - 8)

    def _from_buffer(cls, buffer, offset=0, encoding="utf8"):
        # TODO keep in mind that in windows many funcitons returns wchar encoded in utf16
        return cls._get_data(buffer, offset).decode(encoding).rstrip("\x00")

    def fixed(cls, size):
        if is_integer(size) and size > 0:
            name = f"String{size}"
            data = dict(_size=size)
            return MetaString(name, (String,), data)
        else:
            raise ValueError("String needs a positive integer")


class String(metaclass=MetaString):
    _size = None
    _c_type = "char*"

    def __init__(
        self, string_or_int, _buffer=None, _offset=None, _context=None
    ):
        info = self.__class__._inspect_args(string_or_int)
        size = info.size
        self._buffer, self._offset = allocate_on_buffer(
            size, _context, _buffer, _offset
        )

        self.__class__._to_buffer(
            self._buffer, self._offset, string_or_int, info=info
        )

        self._size = size

    def update(self, value):
        buffer = self._buffer
        offset = self._offset
        size = self._size
        if isinstance(value, String):
            if value._size < self._size:
                buffer.write(offset + 8, value.to_bytes())  # TODO use copy
            else:
                raise ValueError(f"{value} to large to fit in {size}")
        elif isinstance(value, str):
            info = self.__class__._inspect_args(value)
            if info > self._size:
                raise ValueError(f"{value} to large to fit in {size}")
            else:
                data = info.data
                off = _to_slot_size(size) - len(data)
                data += b"\x00" * off
                buffer.write(offset + 8, data)
        else:
            raise ValueError(f"{value} not a string")

    def to_str(self):
        return self.__class__._from_buffer(self._buffer, self._offset)

    def to_bytes(self):
        return self.__class__._get_data(self._buffer, self._offset)

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        return paths


def is_string(atype):
    return atype == String

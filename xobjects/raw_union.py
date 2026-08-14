# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np

from .array import Array
from .scalar import is_scalar
from .typeutils import Info, allocate_on_buffer


class MetaRawUnion(type):
    def __new__(cls, name, bases, data):
        members = {}
        for base in bases:
            members.update(getattr(base, "_members", {}))

        for member_name, member_type in list(data.items()):
            if member_name.startswith("_"):
                continue
            if is_scalar(member_type):
                members[member_name] = member_type
                del data[member_name]

        if members:
            size = max(member._size for member in members.values())
            dtype = np.dtype(f"V{size}")
            data["_members"] = members
            data["_dtype"] = dtype
            data["_size"] = size
            if "_c_type" not in data:
                data["_c_type"] = f"{name}_raw_t"

        return type.__new__(cls, name, bases, data)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)

    def __repr__(cls):
        return cls.__name__


class RawUnion(metaclass=MetaRawUnion):
    _members = {}

    def __init__(self, value=0, _context=None, _buffer=None, _offset=None):
        self._buffer, self._offset = allocate_on_buffer(
            self.__class__._size, _context, _buffer, _offset
        )
        self.__class__._to_buffer(self._buffer, self._offset, value)

    def __getattr__(self, name):
        member_type = self.__class__._members.get(name)
        if member_type is None:
            raise AttributeError(
                f"{self.__class__.__name__} has no RawUnion member {name!r}"
            )
        return member_type._from_buffer(
            self._buffer, self._offset, container=self
        )

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.__class__._members))

    def __repr__(self):
        members = ", ".join(self.__class__._members)
        return f"<{self.__class__.__name__}; members: {members}>"

    @classmethod
    def _member_from_value(cls, value):
        if isinstance(value, tuple):
            member_name, member_value = value
            return cls._members[member_name], member_value
        return next(iter(cls._members.values())), value

    @classmethod
    def _inspect_args(cls, value=0):
        return Info(size=cls._size, value=value)

    @classmethod
    def _from_buffer(cls, buffer, offset=0, container=None):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None, container=None):
        if isinstance(value, cls):
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, cls._size
            )
            return
        member_type, member_value = cls._member_from_value(value)
        data = member_type._dtype.type(member_value).tobytes()
        if len(data) < cls._size:
            data = data + bytes(cls._size - len(data))
        buffer.update_from_buffer(offset, data)

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        return paths


def is_raw_union(cls):
    return isinstance(cls, MetaRawUnion) and cls is not RawUnion

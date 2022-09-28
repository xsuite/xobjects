# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

from .typeutils import allocate_on_buffer, Info
from .scalar import Int64
from .array import Array

"""
union typ1 typ2 ...

Initialization:
    1) Union(instance)
    1) Union((typename,...))


Data layout:
    - typeid
    - data
    - _is_static_type

Array instance:
    _itemtypes: list of item types
"""


class MetaUnion(type):
    def __new__(cls, name, bases, data):
        if "_itemtypes" in data:
            itemtypes = data["_itemtypes"]
            typeids = {}
            types = {}
            isize = itemtypes[0]
            for ii, itemtype in enumerate(itemtypes):
                name = itemtype.__name__
                typeids[name] = ii
                types[name] = itemtype
                if itemtype._size != isize:
                    isize = None

            data["_typeids"] = typeids
            data["_types"] = types
            data["_size"] = isize

        return type.__new__(cls, name, bases, data)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)


class Union(metaclass=MetaUnion):
    _types: list
    _size: None
    _typeids: dict
    _typenames: dict
    _itemtypes: list

    @classmethod
    def _get_type_index(cls, value):
        return cls._typeids[type(value).__name__]

    @classmethod
    def _get_type(cls, name):
        return cls._typenames[name]

    @classmethod
    def add_type(cls, itemtype):
        name = itemtype.__name__
        if name not in cls._typenames:
            cls._itemtypes.append(itemtype)
            cls._types[itemtype.__name__] = itemtype
            cls._typeids[itemtype.__name__] = cls._itemtypes.index(itemtype)
        else:
            raise ValueError(f"{itemtype} already in union")

    @classmethod
    def _inspect_args(cls, *args):
        if len(args) == 1:
            value = args[0]
            if type(value) in cls._itemtypes:
                size = value._get_size() + 8
                return Info(
                    size=size,
                    typeid=cls._get_type_index(value),
                    is_raw=True,
                    value=value,
                )
            elif type(value) is tuple:
                typename, value = value
                itemtype = cls._types[typename]
                typeid = cls._typeids[typename]
                iinfo = itemtype._inspect_args(value)
                return Info(
                    size=iinfo.size + 8,
                    typeid=typeid,
                    extra=iinfo,
                    is_raw=False,
                    value=value,
                )
            else:
                raise ValueError(f"{value} has the wrong type")
        else:
            raise ValueError(f"{value} has wrong number of arguments")

    @classmethod
    def _from_buffer(cls, buffer, offset=0):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        if info is None:
            info = cls._inspect_args(value)
        Int64._to_buffer(buffer, offset, info.typeid)
        coffset = offset + 8
        if info.is_raw:
            info.itemtype._buffer.copy_from(
                coffset, value._buffer, value._offset, info.size
            )
        else:
            info.itemtype._to_buffer(buffer, coffset, value, info=info.extra)

    def __init__(self, *args, _context=None, _buffer=None, _offset=None):
        info = self.__class__._inspect_args(*args)
        self._buffer, self._offset = allocate_on_buffer(
            info.size, _context, _buffer, _offset
        )

        self.__class__._to_buffer(self._buffer, self._offset, info.value, info)

        self._size = info._size
        self._itemtype = info._itemtype

    def get(self):
        return self._itemtype._from_buffer(self._buffer, self._offset + 8)

    def _get_size(self):
        if self.__class__._size is None:
            return Int64._from_buffer(self._buffer, self._offset)
        else:
            return self.__class__._size

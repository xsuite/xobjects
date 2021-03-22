import numpy as np

from .typeutils import get_a_buffer, Info
from .scalar import Int64

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


class MetaUnionRef(type):
    def __new__(cls, name, bases, data):
        if "_itemtypes" in data:
            itemtype = data["_itemtypes"]
            typeids = {}
            types = {}
            for ii, it in enumerate(itemtype):
                name = itemtype.__name__
                typeids[name] = ii
                types[name] = itemtype

            data["_typeids"] = typeids
            data["_types"] = types

        return type.__new__(cls, name, bases, data)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)


class UnionRef(metaclass=MetaUnionRef):
    _size = 16

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
            cls._typesids[itemtype.__name__] = cls._itemtypes.index(itemtype)
        else:
            raise ValueError(f"{itemtype} already in union")

    @classmethod
    def _inspect_args(cls, *args):
        if len(args) == 1:
            value = args[0]
            if type(value) in cls._itemtypes:
                size = value._get_size() + 8
                return Info(
                    isize=size,
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
                    isize=iinfo.size + 8,
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
    def _from_buffer(cls, buffer, offset):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        if info is None:
            info = cls._inspect_args(value)
        Int64._to_buffer(buffer, offset, info.typeid)
        Int64._to_buffer(buffer, offset + 8, info.offset)

    def __init__(self, *args, _context=None, _buffer=None, _offset=None):
        info = self.__class__._inspect_args(*args)
        self._buffer, self._offset = get_a_buffer(_context, _buffer, _offset)

        if info.is_raw and info.value._buffer == self._buffer:
            info.offset = value._offset
        else:
            value = info._itemtype(
                info.value, _buffer=self._buffer, _info=info.extra
            )

        self._itemtype = info._itemtype
        self._value = value

    def get(self):
        return self._value

    def _get_size(self):
        return self._value._get_size()

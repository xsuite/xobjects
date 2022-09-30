# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import logging

import numpy as np

from .typeutils import Info, dispatch_arg, allocate_on_buffer, default_conf
from .scalar import Int64
from .array import Array

log = logging.getLogger(__name__)

NULLVALUE = -(2**63)
NULLTYPE = -1
NULLREF = np.array([NULLVALUE, NULLTYPE], dtype="int64")

# Ref is a like a scalar
class MetaRef(type):
    def __getitem__(cls, reftype):
        return cls(reftype)


class Ref(metaclass=MetaRef):

    _has_refs = True

    def __init__(self, reftype):

        if hasattr(reftype, "_XoStruct"):
            self._reftype = reftype._XoStruct
        else:
            self._reftype = reftype

        self.__name__ = "Ref" + self._reftype.__name__
        self._c_type = self.__name__

        self._size = 8

    def _from_buffer(self, buffer, offset=0):
        refoffset = Int64._from_buffer(buffer, offset)
        if refoffset == NULLVALUE:
            return None
        else:
            refoffset += offset  # from relative to absolute offset
            return self._reftype._from_buffer(buffer, refoffset)

    def _to_buffer(self, buffer, offset, value, info=None):

        # Get/set content
        if value is None:
            refoffset = NULLVALUE  # NULL value
        elif (
            value.__class__.__name__ == self._reftype.__name__  # same type
            and value._buffer is buffer
        ):
            refoffset = value._offset - offset

        else:
            newobj = self._reftype(value, _buffer=buffer)
            refoffset = newobj._offset - offset
        Int64._to_buffer(buffer, offset, refoffset)

    def __call__(self, *args):
        if len(args) == 0:
            return None
        else:
            (value,) = args
            return self._reftype(value)

    def _inspect_args(self, arg):
        return Info(size=self._size)

    def __getitem__(self, shape):
        return Array.mk_arrayclass(self, shape)

    def _gen_data_paths(self, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [self])
        if hasattr(self._reftype, "_gen_data_paths"):
            paths.extend(self._reftype._gen_data_paths(base + [self]))
        return paths

    def _gen_c_decl(self, conf=default_conf):
        from . import capi

        return capi.gen_cdefs(self, [], conf)

    def _gen_c_api(self, conf=default_conf):
        from . import capi

        return capi.gen_code(self, [], conf)

    def __repr__(self):
        return f"<ref {self.__name__}>"

    def _get_inner_types(self):
        return [self._reftype]


# UnionRef is a proper class because
#  - name generation is particularly inconvenient
#  - can be useful to instantiate for debugging
class MetaUnionRef(type):
    _reftypes: list
    _methods: list

    def __new__(cls, name, bases, data):
        if "_c_type" not in data:
            data["_c_type"] = name
        if "_methods" not in data:
            data["_methods"] = []

        data["_has_refs"] = True

        return type.__new__(cls, name, bases, data)

    def _is_member(cls, value):
        typ = value.__class__
        for tt in cls._reftypes:
            if tt.__name__ == typ.__name__:
                return True
        return False

    def _typeid_from_type(cls, typ):
        for ii, tt in enumerate(cls._reftypes):
            if tt.__name__ == typ.__name__:
                return ii
        # If no match found:
        raise TypeError(f"{typ} is not memberof {cls}!")

    def _typeid_from_name(cls, name):
        for ii, tt in enumerate(cls._reftypes):
            if tt.__name__ == name:
                return ii
        # If no match found:
        raise TypeError(f"{name} is not memberof {cls}")

    def _type_from_name(cls, name):
        for tt in cls._reftypes:
            if tt.__name__ == name:
                return tt
        # If no match found:
        raise TypeError(f"{name} is not memberof {cls}")

    def _type_from_typeid(cls, typeid):
        for ii, tt in enumerate(cls._reftypes):
            if ii == typeid:
                return tt
        # If no match found:
        raise TypeError(f"Invalid id: {typeid}!")

    def _from_buffer(cls, buffer, offset=0):
        refoffset, typeid = Int64._array_from_buffer(buffer, offset, 2)
        if refoffset == NULLVALUE:
            return None
        else:
            reftype = cls._type_from_typeid(typeid)
            return reftype._from_buffer(buffer, offset + refoffset)

    def _inspect_args(cls, *args):
        """
        A unionref can be initialized with an instance of the classes in reftypes or
        a tuple (typename, dictionary)

        Input:
        - None
        - XObject
        - typename, dict
        """
        # log.debug(f"get info for {cls} from {args}")
        info = Info(size=cls._size)
        return info

    def _to_buffer(cls, buffer, offset, value, info=None):
        if isinstance(value, cls):  # binary copy
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, value._size
            )
        else:
            if value is None:
                xobj = None
            elif isinstance(value, tuple):
                if len(value) == 0:
                    xobj = None
                    typeid = None
                elif len(value) == 1:  # must be XObject or None
                    xobj = value[0]
                    if xobj is not None:
                        typ = xobj.__class__
                        typeid = cls._typeid_from_type(typ)
                        if xobj._buffer != buffer:
                            xobj = typ(xobj, _buffer=buffer)
                elif len(value) == 2:  # must be (str,dict)
                    tname, data = value
                    typ = cls._type_from_name(tname)
                    typeid = cls._typeid_from_name(tname)
                    xobj = typ(data, _buffer=buffer)
            elif cls._is_member(value):
                xobj = value
                typ = xobj.__class__
                typeid = cls._typeid_from_type(typ)
                if xobj._buffer != buffer:
                    xobj = typ(xobj, _buffer=buffer)
            else:
                raise ValueError(f"{value} not handled")
            if xobj is None:
                Int64._array_to_buffer(buffer, offset, NULLREF)
            else:
                ref = np.array([xobj._offset - offset, typeid])
                Int64._array_to_buffer(buffer, offset, ref)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)

    def _pre_init(cls, *arg, **kwargs):
        return kwargs

    def __repr__(cls):
        return f"<unionref {cls.__name__}>"


class UnionRef(metaclass=MetaUnionRef):
    _size = 16
    _reftypes: list

    def __init__(
        self, *args, _context=None, _buffer=None, _offset=None, **kwargs
    ):
        cls = self.__class__

        args, _ = self._pre_init(*args, **kwargs)

        self._buffer, self._offset = allocate_on_buffer(
            cls._size, _context, _buffer, _offset
        )

        cls._to_buffer(self._buffer, self._offset, args)

        self._post_init()

    def get(self):
        reloffset, typeid = Int64._array_from_buffer(
            self._buffer, self._offset, 2
        )
        if reloffset == NULLVALUE:
            return None
        else:
            cls = self.__class__
            typ = cls._type_from_typeid(typeid)
            offset = self._offset + reloffset
            return typ._from_buffer(self._buffer, offset)

    @classmethod
    def _pre_init(cls, *args, **kwargs):
        return args, kwargs

    def _post_init(self):
        pass

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        return paths

    @classmethod
    def _gen_c_decl(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_cdefs(cls, paths, conf)

    @classmethod
    def _gen_c_api(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_code(cls, paths, conf)

    @classmethod
    def _gen_kernels(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_kernels(cls, paths, conf)

    @classmethod
    def _get_inner_types(cls):
        return cls._reftypes


def is_ref(atype):
    return isinstance(atype, Ref)


def is_unionref(atype):
    return isinstance(atype, MetaUnionRef)

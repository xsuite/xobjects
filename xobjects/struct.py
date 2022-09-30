# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""


struct <name> field1 type1 ... fieldn typen

Layout:
  [ instance size ]: is not static
  static-field1
  ..
  static-fieldn
  [ offset field 2 ]
  [ offset ...
  [ offset field n ]
  [ dynamic-field1 ]
  [ ...
  [ dynamic-fieldn ]

Current implementation:
    1) instance stores the actual offsets for dynamic offset. Although it is wasteful for memory, it avoids a double round trip. This hinges on the structure being frozen at initializations.

Struct class:
- _size: class size, None if not static
- _fields: list of fields
- _d_fields: list of dynamic fields
- _s_fields: list of static fields

Struct instance:
- _offsets: cached offsets of dynamic fields dict indexed by field.index
- _sizes: cached sizes of dynamic fields dict indexed by field.index
- _size: cached size of the object

Field instance:
- ftype
- index
- name
- offset
- is_static_type
- has_update
- readonly


"""
import logging
from typing import Callable, Optional

from .typeutils import (
    allocate_on_buffer,
    dispatch_arg,
    Info,
    _to_slot_size,
    _is_dynamic,
    default_conf,
)

from .scalar import Int64
from .array import Array
from .context import Source

log = logging.getLogger(__name__)


class Field:
    def __init__(
        self, ftype, default=None, readonly=False, default_factory=None
    ):
        self.ftype = ftype
        self.default = default
        self.default_factory = default_factory
        self.index = None  # filled by class creation
        self.name = None  # filled by class creation
        self.offset = None  # filled by class creation
        self.is_reference = None  # filled by class creation
        self.has_update = False  # filled by class creation
        self.readonly = readonly
        self.is_union = None  # filled by class creation

    def __repr__(self):
        return f"<field{self.index} {self.name} at {self.offset}>"

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        else:
            ftype, offset = self.get_offset(instance)
            return ftype._from_buffer(instance._buffer, offset)

    def __set__(self, instance, value):
        """
        A value can be:
          - python value: dict or scalar or list...
          - another instance of the same type
          - ???a buffer_protocol object???
        """
        if self.readonly:
            raise AttributeError(f"Field {self.name} is read-only")

        if hasattr(self.ftype, "_update"):
            self.__get__(instance)._update(value)
        else:  # TODO check if below is really needed
            ftype, offset = self.get_offset(instance)
            ftype._to_buffer(instance._buffer, offset, value)

    def get_offset(self, instance):  # compatible with info
        if self.is_reference:
            reloffset = instance._offsets[self.index]
            if self.is_union:
                absoffset = instance._offset + reloffset
                ftype = self.ftype._get_stored_type(
                    instance._buffer, absoffset
                )
            else:
                ftype = self.ftype
        else:
            reloffset = self.offset
            ftype = self.ftype
        return ftype, instance._offset + reloffset

    def get_default(self):
        if self.default_factory is None:
            if self.default is None:
                return self.ftype()
            else:
                return dispatch_arg(self.ftype, self.default)
        else:
            return self.default_factory()

    def value_from_args(self, arg):
        if self.name in arg:
            return arg[self.name]
        else:
            return self.get_default()


class MetaStruct(type):
    def __new__(cls, name, bases, data):
        offset = 0
        fields = []
        s_fields = []
        d_fields = []
        is_static = True
        findex = 0
        for aname, field in data.items():
            if hasattr(field, "_inspect_args"):
                data[aname] = Field(field)
        for aname, field in data.items():
            if isinstance(field, Field):
                field.index = findex
                findex += 1
                field.name = aname
                if hasattr(field.ftype, "_update"):
                    field.has_update = True
                fields.append(field)
                if field.ftype._size is None:
                    d_fields.append(field)
                    is_static = False
                else:
                    s_fields.append(field)
        data["_fields"] = fields
        data["_s_fields"] = s_fields
        data["_d_fields"] = d_fields

        if is_static:
            offset = 0
            for field in fields:
                field.offset = offset
                field.is_reference = False
                offset += _to_slot_size(field.ftype._size)
            size = offset

            def _get_size(self):
                return self.__class__._size

            def _inspect_args(cls, *args, **kwargs):
                info = Info(size=cls._size, is_static_size=True)
                if len(args) == 1:
                    info.value = args[0]
                else:
                    info.value = kwargs
                return info

        else:
            size = None
            for field in fields:
                offset = 8  # first slot is instance size
                for field in s_fields:
                    field.offset = offset
                    field.is_reference = False
                    offset += _to_slot_size(field.ftype._size)
                # other dynamic fields
                for field in d_fields[1:]:
                    field.offset = offset
                    field.is_reference = True
                    offset += _to_slot_size(8)
                # first dynamic field
                d_fields[0].offset = offset
                d_fields[0].is_reference = False

            def _get_size(self):
                return Int64._from_buffer(self._buffer, self._offset)

            def _inspect_args(cls, *args, **kwargs):
                # log.debug(f"get size for {cls} from {args} {kwargs}")
                info = Info()
                if len(args) == 1:  # is a dict or xobj
                    arg = args[0]
                    info.value = arg
                    if isinstance(arg, dict):
                        offsets = {}  # offset of dynamic data
                        extra = {}
                        offset = d_fields[
                            0
                        ].offset  # offset first dynamic data
                        # log.debug(f"{arg}")
                        for field in cls._d_fields:
                            farg = field.value_from_args(arg)
                            # log.debug(
                            #    f"get size for field `{field.name}` using `{farg}`"
                            # )
                            finfo = dispatch_arg(
                                field.ftype._inspect_args, farg
                            )
                            if hasattr(finfo, "_offsets"):  # is dinamic
                                extra[field.index] = finfo
                            offsets[field.index] = offset
                            offset += _to_slot_size(finfo.size)
                        # _offsets is used to because of field.get_offset(info)
                        info.size = offset
                        info._offsets = offsets
                        if len(extra) > 0:
                            info.extra = extra
                    elif isinstance(arg, cls):
                        info.size = arg._get_size()
                        info._offsets = {
                            kk: vv for kk, vv in arg._offsets.items()
                        }
                    else:
                        raise ValueError(f"{arg} Not valid type for {cls}")
                else:  # python argument
                    return cls._inspect_args(kwargs)
                return info

        data["_size"] = size
        data["_get_size"] = _get_size
        data["_inspect_args"] = classmethod(_inspect_args)
        if "_c_type" not in data:
            data["_c_type"] = name

        # determine owndata
        _has_refs = False
        for ff in data["_fields"]:
            ftype = ff.ftype
            if hasattr(ftype, "_has_refs") and ftype._has_refs:
                _has_refs = True
                break
        data["_has_refs"] = _has_refs
        if "_extra_c_sources" not in data.keys():
            data["_extra_c_sources"] = []
        if "_depends_on" not in data.keys():
            data["_depends_on"] = []
        if "_kernels" not in data.keys():
            data["_kernels"] = {}

        return type.__new__(cls, name, bases, data)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)

    def __repr__(cls):
        return f"<struct {cls.__name__}>"


class Struct(metaclass=MetaStruct):
    _fields: list
    _d_fields: list
    _inspect_args: Callable
    _size: Optional[int]

    @classmethod
    def _pre_init(cls, *args, **kwargs):
        return args, kwargs

    def _post_init(self):
        pass

    @classmethod
    def _from_buffer(cls, buffer, offset=0):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        _offsets = {}
        for field in self._d_fields:
            offset = self._offset + field.offset
            val = Int64._from_buffer(self._buffer, offset)
            _offsets[field.index] = val
        self._offsets = _offsets
        self._size = self._get_size()
        self._post_init()
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        if isinstance(value, cls):  # binary copy
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, value._size
            )
        else:  # value must be a dict, again potential disctructive
            if info is None:
                info = cls._inspect_args(value)
            if cls._size is None:
                Int64._to_buffer(buffer, offset, info.size)
            if hasattr(
                info, "_offsets"
            ):  # if it has a least two dynamic fields
                cls._set_offsets(buffer, offset, info._offsets)
            extra = getattr(info, "extra", {})
            for field in cls._fields:
                fvalue = field.value_from_args(value)
                if field.is_reference:
                    foffset = offset + info._offsets[field.index]
                else:
                    foffset = offset + field.offset
                finfo = extra.get(field.index)
                field.ftype._to_buffer(buffer, foffset, fvalue, finfo)

    def _update(self, value):
        # check if direct copy is possible
        if isinstance(value, self.__class__) and value._size == self._size:
            self._buffer.update_from_xbuffer(
                self._offset, value._buffer, value._offset, value._size
            )
        else:
            for field in self._fields:
                if field.name in value:
                    field.__set__(self, value[field.name])

    def __init__(
        self, *args, _context=None, _buffer=None, _offset=None, **kwargs
    ):
        """
        Create new struct in buffer from offset.
        If offset not provide
        """
        cls = self.__class__
        args, kwargs = cls._pre_init(*args, **kwargs)
        # compute resources
        info = cls._inspect_args(*args, **kwargs)
        self._size = info.size
        # acquire buffer
        self._buffer, self._offset = allocate_on_buffer(
            info.size, _context, _buffer, _offset
        )
        # if dynamic struct store dynamic offsets
        if hasattr(info, "_offsets"):
            self._offsets = info._offsets  # struct offsets
        cls._to_buffer(self._buffer, self._offset, info.value, info)
        self._post_init()

    @classmethod
    def _set_offsets(cls, buffer, offset, loffsets):
        # log.debug(f"{cls} set offset {loffsets}")
        for index, data_offset in loffsets.items():
            foffset = offset + cls._fields[index].offset
            Int64._to_buffer(buffer, foffset, data_offset)

    def _to_dict(self):
        return {field.name: field.__get__(self) for field in self._fields}

    def __iter__(self):
        for field in self._fields:
            yield field.name

    def __getitem__(self, key):
        for field in self._fields:
            if field.name == key:
                return field.__get__(self)
        raise KeyError("{key} not found")

    def __contains__(self, key):
        for field in self._fields:
            if field.name == key:
                return True
        else:
            return False

    def __repr__(self):
        fields = (
            (field.name, repr(getattr(self, field.name)))
            for field in self._fields
        )
        fields = ", ".join(f"{k}={v}" for k, v in fields)
        longform = f"{self.__class__.__name__}({fields})"
        if len(longform) > 60:
            return f"{self.__class__.__name__}(...)"
        else:
            return longform

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        for field in cls._fields:
            path = base + [cls, field]
            paths.append(path)
            if hasattr(field.ftype, "_gen_data_paths"):
                paths.extend(field.ftype._gen_data_paths(path))
        return paths

    @classmethod
    def _gen_c_api(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()

        source = Source(
            source=capi.gen_code(cls, paths, conf),
            name=cls.__name__ + "_gen_c_api",
        )
        return source

    @classmethod
    def _gen_c_decl(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_cdefs(cls, paths, conf)

    @classmethod
    def _gen_kernels(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_kernels(cls, paths, conf)

    def _get_offset(self, fieldname):
        for ff in self._fields:
            if ff.name == fieldname:
                return ff.get_offset(self)[1]

    @classmethod
    def _get_inner_types(cls):
        return [fl.ftype for fl in cls._fields]

    @property
    def _context(self):
        return self._buffer.context

    @classmethod
    def compile_class_kernels(
        cls,
        context,
        only_if_needed=False,
        apply_to_source=(),
        save_source_as=None,
    ):

        if only_if_needed:
            all_found = True
            for kk in cls._kernels.keys():
                if kk not in context.kernels.keys():
                    all_found = False
                    break
            if all_found:
                return

        context.add_kernels(
            sources=[],
            kernels=cls._kernels,
            extra_classes=[cls],
            apply_to_source=apply_to_source,
            save_source_as=save_source_as,
        )

    def compile_kernels(
        self, only_if_needed=False, apply_to_source=(), save_source_as=None
    ):
        self.compile_class_kernels(
            context=self._context,
            only_if_needed=only_if_needed,
            apply_to_source=apply_to_source,
            save_source_as=save_source_as,
        )


def is_struct(atype):
    return isinstance(atype, MetaStruct)


def is_field(atype):
    return isinstance(atype, Field)

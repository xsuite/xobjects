# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
"""A structure type for Xobjects.

The Xobjects structure type can be inherited to implement a collection of
values, both of static (know at C API compile time) and dynamic (known at
initialisation) length.

We define a structure by giving a list of fields, determining the types of
values. For example, consider the following 4-field structure:

    >>> class Example(xo.Struct):
    ...     f0 = xo.UInt8[:]
    ...     f1 = xo.UInt16[2]
    ...     f2 = xo.UInt8[:]
    ...     f3 = xo.UInt8

The fields f1 and f3 are static, as their sizes are know in advance, whereas
the fields f0 and f2 are dynamic, as the amount of memory needed for these
values will only be determined at initialisation.

The memory layout of a general struct is as follows (let N be the number of
static fields, and M the number of dynamic fields):

(1) At offset 0: int64 size of the struct if the struct is dynamic (i.e. has
    dynamic fields); otherwise size is omitted and (2) is here.
(2) At offset 8 (or 0, see above): the N static field values in order, the
    fields are 8-byte aligned.
(3) At the next 8-byte aligned offset: (M - 1) int64 values representing the
    offsets, from the beginning of the struct, of the data corresponding to
    the dynamic fields indexed 1 through (M - 1). Note in particular, that the
    offset of the static field 0 is omitted, as it is implicitly known to start
    immediately after the end of this section.
(4) At the next 8-byte aligned offset: the data corresponding to the dynamic
    field index 0.
(5) At the next 8-byte aligned offset begin, 8-byte aligned, values of dynamic
    fields 1 though (M - 1) pointed to by the offsets specified in (3).
(6) The whole structure is padded, as needed, with zeros to make the size, as
    given in (1), a multiple of 8.

Consider as an instance of the struct `Example` defined above:

    >>> example = Example(f0=[1, 2, 3], f1=[4, 5], f2=[6, 7, 8, 9, 10], f3=11)

Its memory layout can be viewed with:

    >>> example._buffer.buffer.reshape(10, 8)

This reveals the following structure of `example`:

[ 0] 80,  0,  0,  0,  0,  0,  0,  0  # int64 length (here: little endian)
[ 8]  4,  0,  5,  0,  0,  0,  0,  0  # f1: uint16[2] array; 4 bytes padding
[16] 11,  0,  0,  0,  0,  0,  0,  0  # f3: uint8; 7 bytes padding
[24] 56,  0,  0,  0,  0,  0,  0,  0  # int64 field 1 offset (f1 starts at 56)
[32] 24,  0,  0,  0,  0,  0,  0,  0  # f0 header (1/2): int64 size in bytes
[40]  3,  0,  0,  0,  0,  0,  0,  0  # f0 header (2/2): int64 no. of elements
[48]  1,  2,  3,  0,  0,  0,  0,  0  # f0 items: uint8[3]; 5 bytes of padding
[56] 24,  0,  0,  0,  0,  0,  0,  0  # f1 header (1/2): int64 size in bytes
[64]  5,  0,  0,  0,  0,  0,  0,  0  # f1 header (2/2): int64 no. of elements
[72]  6,  7,  8,  9, 10,  0,  0,  0  # f1 items: uint8[5]; 3 bytes of padding
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Type, Tuple

from xobjects.base_type import XoTypeMeta, XoInstanceInfo, XoType, XoType_
from xobjects.typeutils import (
    allocate_on_buffer,
    dispatch_arg,
    _to_slot_size,
    _is_dynamic,
    default_conf,
    is_xo_type,
)
from xobjects.scalar import Int64
from xobjects.context import Source

log = logging.getLogger(__name__)


def is_struct_type(atype):
    return isinstance(atype, MetaStruct)


def is_field(atype):
    return isinstance(atype, Field)


class Field:
    """Represent metadata of a field inside an instance of a Struct.

    Parameters
    ----------
    ftype: Any
        A type of the field, can be an xobjects scalar, array, struct, etc.
    default: instance of ftype
        Initial value of the field; if unspecified the field's value defaults
        to `ftype()`.
    readonly: bool
        Whether the modification of the value is locked after instantiation.
    default_factory: callable
        A callable to be executed in order to create the value of the field.
        If unspecified the field's value defaults to `ftype(default)`.

    Attributes
    ----------
    index: int
        The index of the field within the order of fields in the struct.
    name: str
        The name of the field.
    offset: int
        The offset of the field, in bytes, from the start of the struct.
    is_reference: bool
        A flag indicating if the field is really a pointer. Dynamic fields
        will have a constant `offset` which is a pointer to the actual data
        located later in the struct.
    has_update: bool
        Does the field type implement an _update method, used to update
        the field value.
    is_union: bool
        A flag indicating if the field represents an instance of a union type.
    """
    def __init__(
        self, ftype, default=None, readonly=False, default_factory=None
    ):
        self.ftype = ftype
        self.default = default
        self.default_factory = default_factory
        self.has_update = hasattr(ftype, '_update')
        self.index = None  # filled by class creation
        self.name = None  # filled by class creation
        self.offset = None  # filled by class creation
        self.is_reference = None  # filled by class creation
        self.readonly = readonly
        self.is_union = None  # filled by class creation

    def __repr__(self):
        return f"<field{self.index} {self.name} at {self.offset}>"

    def __get__(self, instance, cls=None):
        """When a field of a struct is accessed, return its value.

        When accessed on the class return its metadata, i.e., the Field object.
        """
        if instance is None:
            return self

        ftype, offset = self.get_offset(instance)
        return ftype._from_buffer(instance._buffer, offset)

    def __set__(self, instance, value):
        """When a field is set on a struct, update the underlying value."""
        if self.readonly:
            raise AttributeError(f"Field {self.name} is read-only")

        if hasattr(self.ftype, "_update"):
            self.__get__(instance)._update(value)
        else:
            ftype, offset = self.get_offset(instance)
            ftype._to_buffer(instance._buffer, offset, value)

    def get_offset(self, instance):  # compatible with info
        """Given an instance of a struct give an offset of the field's data.

        Return an offset, in bytes, where the data of the field begins. This is
        not necessarily the same as `self.offset` for dynamic fields.
        """
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
        """Return the default value of the field."""
        if self.default_factory is not None:
            return self.default_factory()

        if self.default is not None:
            return dispatch_arg(self.ftype, self.default)

        return self.ftype()

    def value_from_args(self, arg):
        if self.name in arg:
            return arg[self.name]

        return self.get_default()

    def get_entry_size(self):
        """Return the size a field takes in the header of a struct."""
        if _is_dynamic(self.ftype):
            return 8
        return self.ftype._size


@dataclass
class StructInstanceInfo(XoInstanceInfo):
    """Metadata representing the allocation requirements of a Struct."""
    is_static_size: bool = False
    value = None
    field_infos = {}
    field_offsets = {}


class MetaStruct(XoTypeMeta):
    """The metaclass for the Xobjects structs."""
    def __new__(mcs, name, bases, data):
        """Create a new struct class.

        Determine the fields of the new struct, and its basic properties (static
        vs dynamic, has a method table, etc.) based on the class definition.
        Generate the methods required by the Struct interface (`_inspect_args`,
        `_get_size`).
        """
        data["_c_type"] = data.get("_c_type", name)

        base_struct = mcs._get_base_struct(bases)
        fields, is_static = mcs._prepare_fields(base_struct, data)

        data["_base_struct"] = base_struct
        data["_fields"] = fields
        data["_is_static"] = is_static

        mcs.assert_inheritance_is_valid(data, base_struct)

        if is_static:
            mcs._make_static_struct_interface(base_struct, data, fields)
        else:
            mcs._make_dynamic_struct_interface(base_struct, data, fields)

        data["_has_refs"] = any(  # TODO: Remove getattr when types are XoType
            getattr(field.ftype, "_has_refs", False) for field in fields
        )

        cls = XoTypeMeta.__new__(mcs, name, bases, data)
        return cls

    @staticmethod
    def _make_dynamic_struct_interface(base_struct, data, fields):
        """Create methods for a dynamic struct."""
        size = None

        if base_struct is not None:
            last_field = base_struct._fields[-1]
            offset = last_field.offset + last_field.get_entry_size()
            new_fields = fields[len(base_struct._fields):]
        else:
            offset = 8
            new_fields = fields

        for field in new_fields:
            field.offset = offset
            if _is_dynamic(field.ftype):
                field.is_reference = True
                field_size = 8  # only the size of the reference
            else:
                field.is_reference = False
                field_size = _to_slot_size(field.ftype._size)

            offset += field_size
        dynamic_section_start = offset

        def _get_size(self):
            """Get the size of a dynamic struct: first 8 bytes."""
            return Int64._from_buffer(self._buffer, self._offset)

        def _inspect_args(cls, xo_or_dict=None, **kwargs):
            extra = {}
            offsets = {}

            if kwargs:
                if xo_or_dict:
                    raise ValueError(
                        'Cannot initialise an object simultaneously from '
                        'kwargs and from a dictionary/xobject.'
                    )
                return cls._inspect_args(kwargs)

            if not xo_or_dict:
                raise ValueError(
                    f'Cannot initialise an instance of {cls.__name__} with '
                    f'{xo_or_dict}.'
                )

            if isinstance(xo_or_dict, dict):
                dict_ = xo_or_dict

                field_offset = dynamic_section_start
                for field in fields:
                    if not _is_dynamic(field.ftype):
                        continue
                    field_arg = field.value_from_args(dict_)
                    field_info = dispatch_arg(field.ftype._inspect_args, field_arg)

                    extra[field.index] = field_info
                    offsets[field.index] = field_offset

                    field_offset += _to_slot_size(field_info.size)

                info = StructInstanceInfo(size=field_offset)
                info.value = xo_or_dict
                info.field_offsets = offsets
                info.field_infos = extra
                return info

            elif isinstance(xo_or_dict, cls):
                instance = xo_or_dict
                info = StructInstanceInfo(size=instance._get_size())
                info.value = xo_or_dict
                info.field_offsets = instance._offsets
                return info

            raise TypeError(
                f'Cannot initialise an instance of {cls} with the argument '
                f'{xo_or_dict} of type {type(xo_or_dict)}.'
            )

        data["_size"] = size
        data["_get_size"] = _get_size
        data["_inspect_args"] = classmethod(_inspect_args)

    @staticmethod
    def _make_static_struct_interface(base_struct, data, fields):
        """Create methods for a static struct."""
        if base_struct is not None:
            offset = base_struct._size
            new_fields = fields[len(base_struct._fields):]
        else:
            offset = 0
            new_fields = fields

        for field in new_fields:
            field.offset = offset
            field.is_reference = False
            offset += _to_slot_size(field.ftype._size)
        size = offset

        def _get_size(_):
            """Get the size of a static struct."""
            return size

        def _inspect_args(_, *args, **kwargs):
            info = StructInstanceInfo(size=size, is_static_size=True)
            if len(args) == 1:
                info.value = args[0]
            else:
                info.value = kwargs
            return info

        data["_size"] = size
        data["_get_size"] = _get_size
        data["_inspect_args"] = classmethod(_inspect_args)

    @staticmethod
    def _prepare_fields(base_struct: Optional[Type['Struct']], data: dict) -> Tuple[List[Field], bool]:
        """Generate Field instances for all typed fields of the class.

        Arguments
        ---------
        base_struct
            The struct the currently processed struct is inheriting from;
            otherwise None.
        data
            The class body.

        Returns
        -------
        fields, is_static
            A list of Fields of the struct, containing metadata about the fields,
            and a boolean indicating whether the struct is static, given the
            information about the fields.
        """
        is_static = True
        fields = []

        if base_struct:
            base_is_static = base_struct._size is not None
            is_static = base_is_static
            fields += base_struct._fields

        field_index = len(fields)

        for field_name, field in data.items():
            if not is_xo_type(field) and not isinstance(field, Field):
                continue

            if hasattr(base_struct, field_name):
                raise TypeError(
                    f"Cannot redeclare the field `{field_name}` as it is already "
                    f"defined in the base struct `{base_struct.__name__}`."
                )

            if is_xo_type(field):
                field = Field(field)

            field.name = field_name
            field.index = field_index

            fields.append(field)
            data[field_name] = field

            if _is_dynamic(field.ftype):
                is_static = False
            field_index += 1

        return fields, is_static

    @staticmethod
    def assert_inheritance_is_valid(data, base_struct):
        if not base_struct:
            return
        if base_struct._is_static != data['_is_static']:
            raise TypeError(
                'A dynamic struct can only inherit from another dynamic struct.'
            )

    def __repr__(cls):
        return f"<struct {cls.__name__}>"

    @staticmethod
    def _get_base_struct(bases) -> Optional[Type['Struct']]:
        base_structs = tuple(
            base for base in bases if is_struct_type(base) and base != Struct
        )

        if base_structs:
            try:
                base_struct, = base_structs
                return base_struct
            except ValueError:
                raise TypeError('Multiple inheritance of xobjects unsupported.')

        return None


class Struct(XoType, metaclass=MetaStruct):
    """Xobject struct type.

    Attributes
    ----------
    _fields:
        List of fields of the struct.
    _base_struct:
        An Xobjects struct type inherited from, or None.
    """

    # Fields filled by the metaclass:
    _fields: List[Field] = []
    _base_struct: Optional[MetaStruct]

    @classmethod
    def _from_buffer(cls, buffer, offset=0):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        offsets = {}
        for field in self._fields:
            if not _is_dynamic(field.ftype):
                continue
            offset = self._offset + field.offset
            val = Int64._from_buffer(self._buffer, offset)
            offsets[field.index] = val

        self._offsets = offsets
        self._size = self._get_size()
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info: StructInstanceInfo = None):
        if isinstance(value, cls) and not cls._has_refs:  # binary copy
            buffer.update_from_xbuffer(
                offset, value._buffer, value._offset, value._size
            )
            return

        # value must be a dict, again potential destructive
        if info is None:
            info = cls._inspect_args(value)

        if _is_dynamic(cls):
            Int64._to_buffer(buffer, offset, info.size)

        if hasattr(info, "field_offsets"):
            cls._set_offsets(buffer, offset, info.field_offsets)

        extra = getattr(info, "field_infos", {})
        for field in cls._fields:
            fvalue = field.value_from_args(value)
            if field.is_reference:
                foffset = offset + info.field_offsets[field.index]
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
        # compute resources
        info = cls._inspect_args(*args, **kwargs)
        self._size = info.size
        # acquire buffer
        self._buffer, self._offset = allocate_on_buffer(
            info.size, _context, _buffer, _offset
        )
        # if dynamic struct store dynamic offsets
        if hasattr(info, "field_offsets"):
            self._offsets = info.field_offsets  # struct offsets
        cls._to_buffer(self._buffer, self._offset, info.value, info)

    @classmethod
    def _set_offsets(cls, buffer, offset, loffsets):
        for index, data_offset in loffsets.items():
            foffset = offset + cls._fields[index].offset
            Int64._to_buffer(buffer, foffset, data_offset)

    def _to_dict(self):
        return {field.name: field.__get__(self) for field in self._fields}

    def _to_json(self):
        out = {}
        for field in self._fields:
            v = field.__get__(self)
            if hasattr(v, "_to_json"):
                v = v._to_json()
            out[field.name] = v
        return out

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

    def __getstate__(self):
        return self._buffer, self._offset

    def __setstate__(self, state):
        self._buffer, self._offset = state

    @classmethod
    def _inspect_args(cls, *args, **kwargs) -> StructInstanceInfo:
        """Determine the allocation requirements of a struct, based on input.

        Implementation of this method is done in the metaclass.

        Arguments
        ---------
        args
            See __init__.
        kwargs
            See __init__.
        """

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
    def _gen_c_api(cls, conf=default_conf) -> Source:
        from . import capi

        paths = cls._gen_data_paths()

        source = capi.gen_code(cls, paths, conf)

        source = Source(
            source=source,
            name=cls.__name__ + "_gen_c_api",
        )
        return source

    @classmethod
    def _gen_c_decl(cls, conf=default_conf) -> str:
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
        dependencies = [fl.ftype for fl in cls._fields]
        if cls._base_struct:
            dependencies.append(cls._base_struct)
        return dependencies

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
        extra_classes=(),
    ):
        if only_if_needed:
            all_found = True
            for kk, kernel_description in cls._kernels.items():
                classes = tuple(kernel_description.get_overridable_classes())
                if (kk, classes) not in context.kernels.keys():
                    all_found = False
                    break
            if all_found:
                return

        context.add_kernels(
            sources=[],
            kernels=cls._kernels,
            extra_classes=[cls] + list(extra_classes),
            apply_to_source=apply_to_source,
            save_source_as=save_source_as,
        )

    def compile_kernels(
        self,
        only_if_needed=False,
        apply_to_source=(),
        save_source_as=None,
        extra_classes=(),
    ):
        self.compile_class_kernels(
            context=self._context,
            only_if_needed=only_if_needed,
            apply_to_source=apply_to_source,
            save_source_as=save_source_as,
            extra_classes=extra_classes,
        )

    @classmethod
    def _get_size(cls, instance: XoType_):
        """Generated by the metaclass."""
        raise RuntimeError(
            'This method should have been generated by the metaclass, but that '
            'has failed for some reason. This indicates a bug.'
        )
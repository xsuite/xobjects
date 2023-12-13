# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
from xobjects.context import XBuffer, Kernel, SourceType

from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, List, Any

XoType_ = TypeVar("XoType_", bound="XoType")


@dataclass
class XoInstanceInfo:
    """Metadata representing the allocation requirements of an XoType."""
    size: int


class XoTypeMeta(ABCMeta):
    def __new__(mcs, name, bases, fields):
        # TODO: Consider if it is a good API choice to "add" to the c sources:
        #  for base in bases:
        #      if not is_xo_type(base):
        #          continue
        #      fields['_extra_c_sources'] += base._extra_c_sources
        #      fields['_depends_on'] += base._depends_on
        #      fields['_kernels'].update(base._kernels)
        return ABCMeta.__new__(mcs, name, bases, fields)

    def __getitem__(cls, shape) -> Type['Array']:
        """Create an array type of `shape`, holding elements of type `cls`.

        Notes
        -----
        The method __getitem__ cannot be made a @classmethod, therefore if we
        want to define it on the type, we need a metaclass. The metaclasses
        of the subclasses of XoType need to inherit XoTypeMeta.
        """
        from xobjects.array import Array  # avoid circular import
        return Array.mk_arrayclass(cls, shape)

    def __repr__(cls):
        """Pretty-print the class name based on its __name__.

        Notes
        -----
        See the remarks on __getitem__.
        """
        return cls.__name__


def is_xo_type(type_var) -> bool:
    """Check that `type_var` is an Xobjects type."""
    return isinstance(type_var, XoTypeMeta)


class XoType(metaclass=XoTypeMeta):
    """A base class for all Xobjects types.

    Properties
    ----------
    _size: int | None
        Size of the instance of the type in bytes. If retrieved as a property
        of the class, `_size` is None if the type is dynamically sized. If
        retrieved as a property of an instance, return size in bytes of the
        instance.
    _c_type: str
        A string representing the type in the autogenerated C code.
    _extra_c_sources: list
        A list of additional sources to be added when compiling the kernels for
        this Xobject.
    _depends_on: list
        A list of Xobjects that the current Xobject depends on in terms of API.
        When building the sources, the sources of the dependencies will be put
        before the sources for the current class.
    _kernels: dict
        Kernel descriptions of the publicly exposed kernels tied to this
        Xobject. A mapping between names and kernel descriptions.
    """
    _size: Optional[int] = None
    _c_type: str
    _extra_c_sources: List[SourceType] = ()
    _depends_on: List[XoType_] = ()
    _kernels: List[str, Kernel] = {}

    @abstractmethod
    def __init__(self):
        """Initialise an instance of XoType.

        Not all types can be instantiated: for example instantiating scalars
        creates numpy scalars.
        """

    @classmethod
    @abstractmethod
    def _inspect_args(cls, *args, **kwargs) -> XoInstanceInfo:
        """Return size and other metadata of a future XoType instance.

        Accepts the same input as __init__, and is used to prepare the input,
        and determine the amount of space that needs to be allocated for an
        xobject.
        """

    @classmethod
    @abstractmethod
    def _get_size(cls, instance: XoType_):
        """Return the size, in bytes, of an instance."""

    @classmethod
    @abstractmethod
    def _from_buffer(cls, buffer: XBuffer, offset=0) -> XoType_:
        """Load an instance of `cls` from `buffer` at `offset`."""

    @classmethod
    def _to_buffer(cls, buffer: XBuffer, offset: int, value, info: XoInstanceInfo = None):
        """Write an instance of `cls` to `buffer` at `offset`

        Arguments
        ---------
        buffer: XBuffer
            A buffer to write to.
        offset: int
            An offset in the buffer at which to write.
        value: Any
            If an instance of `cls`, effectively copy the instance to the offset
            in the buffer. Overlapped copies are not supported and will produce
            undefined results. If an instance of any other object (scalar, dict,
            array), create a new instance of `cls` in the described location in
            the buffer using the provided `value` as input to the initializer.
        info: XoInstanceInfo
            In the case where `value` is used as initializer input, if any
            metadata useful for object allocation is already known, it can be
            provided as `info` to speed up the allocation (skip a call to
            `_inspect_args`).
        """

    @classmethod
    @abstractmethod
    def _gen_data_paths(cls, base: List[XoTypeMeta] = None) -> List[List[Any]]:
        """Return a list of paths to each field in the class hierarchy.

        Returns
        -------
        List[List[Any]]
            Return a list, in which for every field in the class (and the fields
            of the fields, and so on) there is a corresponding element, a list,
            representing a path from the class, through the intermediate fields,
            to the field. Used to generate the helper functions (getters,
            setters, etc.) of the C API.
        """
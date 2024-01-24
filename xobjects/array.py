# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
"""An array type for Xobjects.

There are fundamentally 2 types of arrays w.r.t. the shape: static (where the
dimensions of the array are predetermined at class creation), and dynamic
(where some/all the dimensions of the array are not determined at class
creation, and instead only on the instantiation of the type).

In the case of a dynamically shaped array the actual sizes of the dimensions
are stored in the metadata of the array. For a static shaped array this is not
needed as the information is derived from the type.

An array stores elements of the same Xobject type (note that compound types are
also allowed). Based on the element there are further two classifications of
the arrays based on the item type: static or dynamic.

An array which holds static elements will store the data directly within the
data slots (as their sizes are known and uniform). An array storing
dynamic-typed elements will instead store references (in the form of offsets)
to a further point in the buffer allocated for the array where the items reside,
similarly to the implementation of an Xobjects struct type.

The memory layout of an Xobjects array is as follows:

(1) At offset 0: int64 size of the struct if the array has dynamic fields or
    has a dynamic shape (i.e. its size is variable); otherwise size is omitted
    and (2) is here.
(2) If the array has a dynamic shape, N * int64 values corresponding to the
    sizes of the N dynamic dimensions.
(3) If the array has a dynamic shape with more than one dimension, M * int64
    values corresponding to the strides, where M is the number of dimensions.
(4) If item type is static, a contiguous list of elements in the order according
    to the strides. If item type is dynamic, a contiguous list of references
    (offsets) to the elements located further in the buffer.

Refer to the documentation of `Array.__init__` for valid ways of instantiating
an Xobjects array, and of `Array.make_array_class` for information on how to
create Xobjects array types (with custom shapes, ordering/strides, etc.).

Examples of memory layouts:

    >>> Arr2xNx2xMUint8 = xo.UInt8[2, :, 2, :]
    >>> mix = Arr2xNx2xMUint8(np.arange(48).reshape(2, 3, 2, 4))
    >>> mix._buffer.buffer

    array([104,   0,   0,   0,   0,   0,   0,   0,  # size in bytes (int64)
             3,   0,   0,   0,   0,   0,   0,   0,  # 1st dyn. dim. size (int64)
             4,   0,   0,   0,   0,   0,   0,   0,  # 2nd dyn. dim. size (int64)
            24,   0,   0,   0,   0,   0,   0,   0,  # 1st strides (int64)
             8,   0,   0,   0,   0,   0,   0,   0,  # 2nd ditto
             4,   0,   0,   0,   0,   0,   0,   0,  # 3rd ditto
             1,   0,   0,   0,   0,   0,   0,   0,  # 4th ditto
             0,   1,   2,   3,   4,   5,   6,   7,  # contents of the array
             8,   9,  10,  11,  12,  13,  14,  15,  #  ordered according to the
            16,  17,  18,  19,  20,  21,  22,  23,  #  strides (in this case
            24,  25,  26,  27,  28,  29,  30,  31,  #  usual C ordering is
            32,  33,  34,  35,  36,  37,  38,  39,  #  applied;
            40,  41,  42,  43,  44,  45,  46,  47], #  48 uint8 values)
          dtype=int8)

    >>> SimpleF = xo.UInt8[3:2, 3:1, 3:0]
    >>> simple_f = SimpleF(np.arange(27).reshape(3, 3, 3).tolist())
    >>> simple_f._buffer.buffer

    array([ 0,  9, 18,  3, 12, 21,  6, 15,  # values arranges according to
           24,  1, 10, 19,  4, 13, 22,  7,  # the Fortran ordering, i.e.
           16, 25,  2, 11, 20,  5, 14, 23,  # _order=(2, 1, 0), and so _strides=
            8, 17, 26,  0,  0,  0,  0,  0], # (1, 3, 9); 5 bytes padding
          dtype=int8)
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union, Literal

import numpy as np

from .base_type import XoTypeMeta, XoType, XoInstanceInfo
from .typeutils import (
    allocate_on_buffer,
    is_integer,
    _to_slot_size,
    default_conf,
)
from .scalar import Int64, is_scalar

log = logging.getLogger(__name__)


def get_suffix(shape):
    # write suffix  6x6x6 or Nx6xM
    sshape = []
    ilst = 0
    lst = "NMOPQRSTUVWXYZABCDEFGHIJKLM"
    for dd in shape:
        if dd is None:
            sshape.append(lst[ilst])
            ilst = (ilst + 1) % len(lst)
        else:
            sshape.append(str(dd))
    return "x".join(sshape)


def get_shape_from_array(value, num_dimensions: int):
    """Deduce the shape of the `value`, considering at most `num_dimensions`.

    Arguments
    ---------
    value
        Array-type argument, whose shape is to be determined.
    num_dimensions
        The depth, or the maximum number of dimensions to be considered when
        determining the shape of `value`.
    """
    if hasattr(value, "shape"):
        return value.shape
    if hasattr(value, "_shape"):
        return value._shape
    if hasattr(value, "lower"):  # test for string, bytes, bytearray, ...
        return ()
    if hasattr(value, "__len__"):
        shape = (len(value),)
        if len(value) > 0 and num_dimensions > 1:
            shape0 = get_shape_from_array(value[0], num_dimensions - 1)
            if shape0 == ():
                return shape

            for i in value[1:]:
                shape_i = get_shape_from_array(i, num_dimensions - 1)
                if shape_i != shape0:
                    raise ValueError(
                        f"{value} does not have a consistent shape in "
                        f"dimension {num_dimensions}"
                    )
            return shape + shape0
        return shape
    return ()


def get_strides(shape, order, itemsize):
    """Compute the byte distance between consecutive elements in each dimension.

    The offset of the element with index `idx` can be calculated as:

        offset = strides[0] * idx[0] + strides[1] * idx[1] + ... + strides[n] * idx[n]

    Arguments
    ---------
    shape
        Dimension for each index.
    order
        Order of the dimensions in the memory layout. 0 is the slowest variation.
    itemsize
        Size of a single element of the array in bytes.

    Return
    ------
    Tuple[int]
        Strides for each index.
    """
    cshape = [shape[io] for io in order]
    cstrides = get_c_strides(cshape, itemsize)
    return tuple(cstrides[order.index(ii)] for ii in range(len(order)))


def get_c_strides(shape, itemsize):
    """Calculate strides assuming C ordering."""
    ss = itemsize
    strides = []
    for sh in reversed(shape):
        strides.append(ss)
        ss *= sh
    return tuple(reversed(strides))


def iter_index(shape, order):
    """Return index in order of data layout"""
    if len(shape) == 1:
        yield from range(shape[0])
    else:
        flipped_order = [order.index(ii) for ii in range(len(order))]
        for ii in np.ndindex(*[shape[io] for io in order]):
            yield tuple(ii[io] for io in flipped_order)


def mk_order(order: Union[Literal['C', 'F'], Tuple[int]], shape: Tuple[int]) -> Tuple[int]:
    """Make the order (list) of dimensions.

    If order is 'C' or 'F', return a C-ordering or a Fortran-ordering,
    respectively, based on `shape`. Otherwise, just return the input `order`.
    """
    if order == "C":
        return tuple(range(len(shape)))
    elif order == "F":
        return tuple(range(len(shape) - 1, -1, -1))
    else:
        return order


def get_offset(idx, strides):
    """Calculate the offset of element `idx` given `strides`."""
    return sum(ii * ss for ii, ss in zip(idx, strides))


def bound_check(index, shape):
    """Check that `index` is a valid index in an array with `shape`."""
    if not all((0 <= ii < max_ii) for ii, max_ii in zip(index, shape)):
        raise IndexError(f"Index {index} outside shape {shape}")


class Index:
    def __init__(self, cls):
        self.cls = cls


@dataclass
class ArrayInstanceInfo(XoInstanceInfo):
    """Metadata representing the allocation requirements of an Array."""
    items = None
    value = None
    extra = {}
    offsets = {}
    shape = None
    strides = None
    order = None


class MetaArray(XoTypeMeta):
    """The metaclass for an Xobjects array type."""
    def __new__(mcs, name, bases, data):
        """Create a new array class.

        Determine the fields (variables and methods) of the new array, and
        its basic properties (static vs dynamic, if dynamic, the shape and
        strides, etc.) based on the class definition. Generate the methods
        required by the Array interface (`_inspect_args`, `_get_size`).
        """
        if "_c_type" not in data:
            data["_c_type"] = name

        if "_itemtype" not in data:
            # This branch is only expected to be taken for the 'Array' class
            # below (or any other custom array class implementing this metaclass
            # directly in the future). For all other cases we go through
            # 'Array.make_array_class', which specifies '_itemtype' in the body.
            return XoTypeMeta.__new__(mcs, name, bases, data)

        return mcs.new_with_itemtype(name, bases, data)

    @classmethod
    def new_with_itemtype(mcs, name, bases, data):
        """Create a new array class when item type is provided in the body."""
        data_offset = 0
        itemtype = data["_itemtype"]

        if itemtype._size is None:
            static_size = 8
            data["_is_static_type"] = False
        else:
            static_size = itemtype._size
            data["_is_static_type"] = True

        if "_shape" not in data:
            raise ValueError(f"No shape defined for {mcs}")
        shape = data["_shape"]

        dshape = []  # find dynamic shapes
        for ii, d in enumerate(shape):
            if d is None:
                data["_is_static_shape"] = False
                dshape.append(ii)

        if len(dshape) > 0:
            data["_dshape_idx"] = dshape
            data_offset += len(dshape) * 8  # space for dynamic shapes
            if len(shape) > 1:
                data_offset += len(shape) * 8  # space for strides
            else:
                data["_strides"] = (static_size,)
        else:
            data["_is_static_shape"] = True
            data["_order"] = mk_order(data["_order"], shape)
            data["_strides"] = get_strides(
                shape, data["_order"], static_size
            )

        if data["_is_static_shape"] and data["_is_static_type"]:
            _size = itemtype._size
            for d in shape:
                _size *= d
            _size = _to_slot_size(_size)
        else:
            _size = None
            data_offset += 8  # space for dynamic size
        data["_size"] = _size
        data["_data_offset"] = data_offset

        # TODO: Remove getattr once all xo types inherit XoType:
        data["_has_refs"] = getattr(data["_itemtype"], '_has_refs', False)

        return XoTypeMeta.__new__(mcs, name, bases, data)

    def _get_offset(cls, index):
        return get_offset(index, cls._strides)

    def _get_n_items(cls):
        if cls._is_static_shape:
            return np.prod(cls._shape)
        else:
            raise ValueError("Cannot get n items from dynamic shapes")

    def __repr__(cls):
        return f"<array {cls.__name__}>"


ShapeWithStrides = Union[int, slice, Tuple[Union[int, slice, None], ...]]


class Array(XoType, metaclass=MetaArray):
    """Xobjects array type.

    Attributes
    ----------
    _shape
        The shape of the array (as accessed, but not necessarily in memory, see
        `_order` and `_strides`). A tuple of ints representing the size of the
        corresponding dimension, or None if the dimension is dynamically sized.
        In an instance `_shape` contains no `None`s and has the actual shape
        of the allocated array.
    _order
        A tuple of ints in `range(len(_shape))` representing the order of
        dimensions in memory. If "C", it defaults to the C ordering (0, ..., N),
        and if "F" it defaults to the Fortran ordering (N, ..., 0), where N is
        the number of dimensions.
    _strides
        A tuple of ints. The i-th entry in the tuple is the distance in bytes
        between two elements whose indices only differ by one in the i-th
        dimension. Available in a class if the shape is static, otherwise
        only a property of an instance.
    _itemtype
        The Xobject type contained by the array.
    _is_static_shape
        True iff the shape has no dynamic dimensions.
    _is_static_type
        True iff `_itemtype` has a static size.
    _data_offset
        The offset in bytes to the first data entry (or a pointer).
    _dshape_idx
        Indices of dynamic shapes.
    _dshape
        In an instance, the sizes of the dynamic dimensions.
    """
    _shape: Tuple[Union[int, None]]
    _order: Union[Literal["C", "F"], Tuple[int]] = "C"
    _strides: Tuple[int]
    _itemtype: XoTypeMeta
    _is_static_shape: bool
    _is_static_type: bool
    _data_offset: int
    _dshape_idx: Tuple[int]

    _dshape: Tuple[int]

    @classmethod
    def make_array_class(cls, itemtype: XoTypeMeta, shape: ShapeWithStrides) -> MetaArray:
        """Create an Xobjects array class for `itemtype` with `shape`.

        Arguments
        ---------
        itemtype
            The Xobjects type of the elements contained by the array.
        shape
            The desired shape of the array given as a tuple of elements each
            specifying the size of the respective dimension. A non-tuple shape
            is presumed to be a single size of the only dimension of the array.
            If the n-th element of the tuple is an integer, this simply
            describes the size of the n-th dimension, if it is a slice, `start`
            of the slice is the size of the n-th dimension, whereas `stop` is
            the value used to compute custom strides, and which represents the
            actual dimension in the memory layout of the given (n-th) dimension.
            If the value representing the size is None, then the corresponding
            dimension is taken to be variable-size (dynamic).

            In other words, an array with shape=(2, 2, 2) is a simple 2x2x2
            C-style array (equivalent to one with shape=(2:0, 2:1, 2:2)),
            whereas an array with shape=(2:2, 2:1, 2:0) would be an array with
            a Fortran-style memory layout. Note that addressing the elements in
            either case is the same, and only the underlying memory order is
            different. Nevertheless, specifying custom strides can be useful for
            interfacing with Fortran code, or for performance reasons to ensure
            certain spacial locality.
        """
        if type(shape) in (int, slice):
            shape = (shape,)
        order = list(range(len(shape)))
        nshape = []
        for ii, dd in enumerate(shape):
            if type(dd) is slice:
                nshape.append(dd.start)
                if dd.stop is not None:
                    order[ii] = dd.stop
            else:
                nshape.append(dd)

        suffix = get_suffix(nshape)

        name = f"Arr{suffix}{itemtype.__name__}"

        data = {
            "_itemtype": itemtype,
            "_shape": tuple(nshape),
            "_order": tuple(order),
        }
        return MetaArray(name, (cls,), data)

    @classmethod
    def _inspect_args(cls, *args) -> ArrayInstanceInfo:
        """
        Determine:
        - size:
        - shape, order, strides
        - offsets
        - value: None if args contains dimensions else args[0]
        """
        # log.debug(f"get size for {cls} from {args}")
        extra = {}
        offsets = None
        dshape = None

        if cls._size is not None:
            # Array of predetermined shape with static items: not much to do
            # but check the passed input is compatible.
            if len(args) == 0:
                value = None
            elif len(args) == 1:
                shape = get_shape_from_array(args[0], len(cls._shape))
                if shape != cls._shape:
                    raise ValueError(
                        f"Cannot initialise {cls.__name__} with {args[0]}, as "
                        f"its shape is not compatible."
                    )
                value = args[0]
            elif len(args) > 1:
                raise ValueError("too many arguments")
            return cls._inspect_args_static(value)
        else:  # handling other cases
            offset = 8  # space for size data
            # determine shape and order
            if cls._is_static_shape:
                shape = cls._shape
                order = cls._order
                strides = cls._strides
                items = np.prod(shape)
                value = args[0]
            else:  # complete dimensions
                if len(args) == 0:
                    raise ValueError(
                        "Cannot initialize array with dynamic shape without arguments"
                    )
                if not is_integer(args[0]):  # init with array
                    value = args[0]
                    shape = get_shape_from_array(value, len(cls._shape))
                    dshape = []
                    for idim, ndim in enumerate(cls._shape):
                        if ndim is None:
                            dshape.append(idim)
                        else:
                            if shape[idim] != ndim:
                                raise ValueError(
                                    "Array: incompatible dimensions"
                                )
                    value = value
                else:  # init with shapes
                    if cls._itemtype._size is None:
                        raise (
                            ValueError(
                                "Cannot initialize a dynamic array with a dynamic type using length"
                            )
                        )
                    value = None
                    shape = []
                    dshape = []  # index of dynamic shapes
                    for ndim in cls._shape:
                        if ndim is None:
                            shape.append(args[len(dshape)])
                            dshape.append(len(shape))
                        else:
                            shape.append(ndim)
                # now we have shape, dshape
                shape = shape
                offset += len(dshape) * 8  # space for dynamic shapes
                if len(shape) > 1:
                    offset += len(shape) * 8  # space for strides
                order = mk_order(cls._order, shape)
                if cls._is_static_type:
                    strides = get_strides(shape, order, cls._itemtype._size)
                else:
                    strides = get_strides(shape, order, 8)
                items = np.prod(shape)

            # needs items, order, shape, value
            if cls._is_static_type:
                # offsets = np.empty(shape, dtype="int64")
                # for idx in iter_index(shape, order):
                #    offsets[idx] = offset
                #    offset += cls._itemtype._size
                offset += cls._itemtype._size * items
                size = _to_slot_size(offset)

            else:
                # args must be an array of correct dimensions
                offsets = np.empty(shape, dtype="int64")
                offset += items * 8
                for idx in iter_index(shape, order):
                    extra[idx] = cls._itemtype._inspect_args(value[idx])
                    offsets[idx] = offset
                    offset += extra[idx].size
                size = _to_slot_size(offset)

            info = ArrayInstanceInfo(size=size)
            info.shape = shape
            info.strides = strides
            info.order = order
            info.value = value
            info.offsets = offsets
            info.extra = extra
            info.dshape = dshape
            info.items = np.prod(shape)
            return info

    @classmethod
    def _inspect_args_static(cls, value):
        offsets = np.empty(cls._shape, dtype="int64")
        offset = 0
        for idx in iter_index(cls._shape, cls._order):
            offsets[idx] = offset
            offset += cls._itemtype._size

        assert cls._size == _to_slot_size(offset)

        info = ArrayInstanceInfo(size=cls._size)
        info.shape = cls._shape
        info.strides = cls._strides
        info.order = cls._order
        info.value = value
        info.offsets = offsets
        info.items = np.prod(cls._shape)
        return info

    @classmethod
    def _from_buffer(cls, buffer, offset=0):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        coffset = offset
        if cls._size is None:
            self._size = Int64._from_buffer(self._buffer, coffset)
            coffset += 8

        if cls._is_static_shape:
            shape = cls._shape
        else:
            shape = []
            for dd in cls._shape:
                if dd is None:
                    shape.append(Int64._from_buffer(self._buffer, coffset))
                    coffset += 8
                else:
                    shape.append(dd)
            self._shape = shape
            if len(shape) > 1:  # getting strides
                # could be computed from shape and order but offset needs to taken
                strides = []
                for _ in range(len(shape)):
                    strides.append(Int64._from_buffer(self._buffer, coffset))
                    coffset += 8
            else:
                if cls._is_static_type:
                    strides = (cls._itemtype._size,)
                else:
                    strides = (8,)
            self._strides = tuple(strides)

        if not cls._is_static_type:
            items = np.prod(shape)
            self._offsets = Int64._array_from_buffer(buffer, coffset, items)

        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        if info is None:
            info = cls._inspect_args(value)
        value = info.value  # can be None if value contained shape info
        header = []
        coffset = offset

        if cls._size is None:
            header.append(info.size)

        if not cls._is_static_shape:
            for ii, nd in enumerate(cls._shape):
                if nd is None:
                    header.append(info.shape[ii])
            if len(cls._shape) > 1:
                header.extend(info.strides)
        if len(header) > 0:
            Int64._array_to_buffer(
                buffer, coffset, np.array(header, dtype="i8")
            )
            coffset += 8 * len(header)
        if not cls._is_static_type:
            Int64._array_to_buffer(buffer, coffset, info.offsets)
            coffset += 8 * len(info.offsets)

        if (
                hasattr(cls._itemtype, "_dtype") and
                hasattr(value, "dtype") and
                value.strides == getattr(cls, '_strides', None)
        ):  # is a numpy array of the same layout
            if not isinstance(value, buffer.context.nplike_array_type):
                value = buffer.context.nparray_to_context_array(value)
            buffer.update_from_nplike(coffset, cls._itemtype._dtype, value)
        elif isinstance(value, cls):
            if value._size == info.size:
                buffer.update_from_xbuffer(
                    offset, value._buffer, value._offset, value._size
                )
            else:
                raise ValueError("Value {value} not compatible size")
        elif value is None:  # no value to initialize
            if is_scalar(cls._itemtype):
                pass  # leave uninitialized
            else:
                value = cls._itemtype()  # use default type
                if cls._is_static_type:
                    ioffset = offset + cls._data_offset
                    for idx in range(info.items):
                        cls._itemtype._to_buffer(buffer, ioffset, value, None)
                        ioffset += cls._itemtype._size
                else:
                    for idx in iter_index(info.shape, cls._order):
                        cls._itemtype._to_buffer(
                            buffer,
                            offset + info.offsets[idx],
                            value[idx],
                            info.extra.get(idx),
                        )
        else:  # there is a value for initialization
            if not hasattr(value, "shape"):  # not nplike
                value = np.asarray(value, dtype=object)
            if cls._is_static_type:
                ioffset = offset + cls._data_offset
                for idx in iter_index(info.shape, cls._order):
                    cls._itemtype._to_buffer(
                        buffer, ioffset, value[idx], info=None
                    )
                    ioffset += cls._itemtype._size
            else:
                for idx in iter_index(info.shape, cls._order):
                    cls._itemtype._to_buffer(
                        buffer,
                        offset + info.offsets[idx],
                        value[idx],
                        info.extra.get(idx),
                    )

    def __init__(self, *args, _context=None, _buffer=None, _offset=None):
        """Instantiate an Xobjects array.

        Arguments
        ---------
        *args
            Values of *args determine the contents to initialise the array with:
            - If `_size` is not None (the array is not dynamic), then calling
              with `args=()` will instantiate a zero filled array.
            - If the array is dynamically shaped, calling with `args=(n_0, n_1,
              ..., n_N)`, where N is the number of dynamic dimensions, will
              instantiate a zero filled array where the shape is as determined
              by _shape (for the static dimensions) and n_0, n_1, ..., n_N for
              the dynamic dimensions.
            - Calling with `args=array`, where `array` is a list or numpy array,
              will instantiate an Xobjects array of an appropriate size
              containing the elements of array.
        _context
            The target Xobjects context to allocate the array.
        _buffer
            The target Xobjects buffer to allocate the array.
        _offset
            The offset in the `buffer` at which the array will be allocated.
        """
        cls = self.__class__
        info = cls._inspect_args(*args)

        self._buffer, self._offset = allocate_on_buffer(
            info.size, _context, _buffer, _offset
        )

        cls._to_buffer(self._buffer, self._offset, info.value, info)

        if cls._size is None:
            self._size = info.size
        if not cls._is_static_shape:
            self._shape = info.shape
            self._dshape = info.dshape
            self._strides = info.strides
        if not cls._is_static_type:
            self._offsets = info.offsets

    @classmethod
    def _get_size(cls, instance):
        if cls._size is None:
            return Int64._from_buffer(instance._buffer, instance._offset)
        else:
            return cls._size

    @classmethod
    def _get_position(cls, index):
        offset = get_offset(index, cls._strides)
        if cls._is_static_type:
            return offset // 8
        else:
            return offset // cls._itemtype._size

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            index = (index,)
        cls = self.__class__
        if hasattr(self, "_offsets"):
            offset = self._offset + self._offsets[index]
        else:
            bound_check(index, self._shape)
            offset = (
                self._offset
                + cls._data_offset
                + get_offset(index, self._strides)
            )
        return cls._itemtype._from_buffer(self._buffer, offset)

    def __setitem__(self, index, value):
        if isinstance(index, (int, np.integer)):
            index = (index,)
        cls = self.__class__
        if hasattr(cls._itemtype, "_update"):
            self[index]._update(value)
        else:
            if hasattr(self, "_offsets"):
                offset = self._offset + self._offsets[index]
            else:
                bound_check(index, self._shape)
                offset = (
                    self._offset
                    + cls._data_offset
                    + get_offset(index, self._strides)
                )
            cls._itemtype._to_buffer(self._buffer, offset, value)

    def _update(self, value):
        if is_integer(value):
            ll = value
        else:
            ll = len(value)
        if len(self) == ll:
            self.__class__._to_buffer(self._buffer, self._offset, value)
        else:
            if is_integer(value):
                raise ValueError(f"Cannot specify new length {ll} for {self}")
            else:
                raise ValueError(
                    f"len({value})={ll} is incompatible with len({self})={len(self)}"
                )

    def _get_offset(self, index):
        if isinstance(index, (int, np.integer)):
            index = (index,)
        cls = self.__class__
        if hasattr(self, "_offsets"):
            offset = self._offset + self._offsets[index]
        else:
            bound_check(index, self._shape)
            offset = (
                self._offset
                + cls._data_offset
                + get_offset(index, self._strides)
            )
        return offset

    def _iter_index(self):
        return iter_index(self._shape, self._order)

    def __len__(self):
        return np.prod(self._shape)

    def to_nplike(self):
        shape = self._shape
        cshape = [shape[ii] for ii in self._order]
        if hasattr(self._itemtype, "_dtype"):
            arr = self._buffer.to_nplike(
                self._offset + self._data_offset, self._itemtype._dtype, cshape
            ).transpose(self._order)
            assert arr.strides == self._strides
            return arr
        else:
            raise NotImplementedError

    def to_nparray(self):
        shape = self._shape
        cshape = [shape[ii] for ii in self._order]
        if hasattr(self._itemtype, "_dtype"):
            arr = self._buffer.to_nparray(
                self._offset + self._data_offset, self._itemtype._dtype, cshape
            ).transpose(self._order)
            assert arr.strides == self._strides
            return arr
        else:
            raise NotImplementedError

    @classmethod
    def _gen_data_paths(cls, base=None):
        paths = []
        if base is None:
            base = []
        paths.append(base + [cls])
        path = base + [cls, Index(cls)]
        paths.append(path)
        if hasattr(cls._itemtype, "_gen_data_paths"):
            paths.extend(cls._itemtype._gen_data_paths(path))
        return paths

    @classmethod
    def _gen_c_api(cls, conf=default_conf):
        from . import capi

        paths = cls._gen_data_paths()
        return capi.gen_code(cls, paths, conf)

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

    def __repr__(self):
        return f"<{self.__class__.__name__}{self._shape} at {self._offset}>"

    @classmethod
    def _get_inner_types(cls):
        return [cls._itemtype]

    def _to_json(self):
        out = []
        for v in self:  # TODO does not support multidimensional arrays
            if hasattr(v, "_to_json"):
                vdata = v._to_json()
            else:
                vdata = v
            if self._has_refs and v is not None:
                vdata = (v.__class__.__name__, vdata)
            out.append(vdata)
        return out


def is_index(atype):
    return isinstance(atype, Index)


def is_array(atype):
    return isinstance(atype, MetaArray)

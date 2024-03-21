# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import logging

import numpy as np


from .typeutils import (
    allocate_on_buffer,
    Info,
    is_integer,
    _to_slot_size,
    default_conf,
)
from .scalar import Int64, is_scalar

log = logging.getLogger(__name__)


"""
array itemtype d1 d2 ...
array itemtype d1 : ...
array itemtype (d1,1) (d1,0) ...  F contiguos

There ara 4 kind of arrays from the combination of
    shape: static, dynamic
    item: static, dynamic


Data layout:
    - [size]: if not (static,static)
    - [d0 d1 ...]: dynamic dimensions (dynamic,*)
    - [stride1 stride2 ...] if nd>1
    - [offsets]: if itemtype is not static (*|dynamic)
    - data: array data

Array class:
    - _size
    - _shape: the shape in the index space
    - _dshape_idx: index of dynamic shapes
    - _order: the hierarchical order of the indexes
    - _itemtype
    - _is_static_shape
    - _strides
    - _is_static_type

Array instance:
    - _dshape: value of dynamic dimensions
    - _shape: present if dynamic
    - _strides: shape if dynamic

Initialization:

Arr(): if _size is not None
Arr(d1,d2,...): using dimensions if _itemtype._size is not None
Arr(arr): if arr is a list, tuple or np-like
"""


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


def get_shape_from_array(value, nd):
    if hasattr(value, "shape"):
        return value.shape
    elif hasattr(value, "_shape"):
        return value._shape
    if hasattr(value, "lower"):  # test for string
        return ()
    elif hasattr(value, "__len__"):
        shape = (len(value),)
        if len(value) > 0 and nd > 1:
            shape0 = get_shape_from_array(value[0], nd - 1)
            if shape0 == ():
                return shape
            for i in value[1:]:
                shapei = get_shape_from_array(i, nd - 1)
                if shapei != shape0:
                    raise ValueError(
                        f"{value} does not have a "
                        f"consistent shape in dimension {nd}"
                    )
            return shape + shape0
        else:
            return shape
    else:
        return ()


def get_strides(shape, order, itemsize):
    """
    shape dimension for each index
    order of the dimensions in the memory layout
    - 0 is slowest variation
    return strides for each index

    off=strides[0]*idx[0]+strides[1]*idx[1]+...+strides[n]*idx[n]

    """
    cshape = [shape[io] for io in order]
    cstrides = get_c_strides(cshape, itemsize)
    return tuple(cstrides[order.index(ii)] for ii in range(len(order)))


def get_f_strides(shape, itemsize):
    """
    calculate strides assuming F ordering
    """
    ss = itemsize
    strides = []
    for sh in shape:
        strides.append(ss)
        ss *= sh
    return tuple(strides)


def get_c_strides(shape, itemsize):
    """
    calculate strides assuming C ordering
    """
    ss = itemsize
    strides = []
    for sh in reversed(shape):
        strides.append(ss)
        ss *= sh
    return tuple(reversed(strides))


def iter_index(shape, order):
    """return index in order of data layout"""
    if len(shape) == 1:
        for ii in range(shape[0]):
            yield ii
    else:
        aorder = [order.index(ii) for ii in range(len(order))]
        for ii in np.ndindex(*[shape[io] for io in order]):
            yield tuple(ii[io] for io in aorder)


def mk_order(order, shape):
    if order == "C":
        return list(range(len(shape)))
    elif order == "F":
        return list(range(len(shape) - 1, -1, -1))
    else:
        return order


def get_offset(idx, strides):
    return sum(ii * ss for ii, ss in zip(idx, strides))


def bound_check(index, shape):
    for ii, ss in zip(index, shape):
        if ii < 0 or ii >= ss:
            raise IndexError(f"index {index} outside shape {shape}")


class Index:
    def __init__(self, cls):
        self.cls = cls


class MetaArray(type):
    def __new__(cls, name, bases, data):
        if "_itemtype" in data:  # specialized class
            _data_offset = 0
            _itemtype = data["_itemtype"]
            if _itemtype._size is None:
                static_size = 8
                data["_is_static_type"] = False
            else:
                data["_is_static_type"] = True
                static_size = _itemtype._size
            if "_shape" not in data:
                raise ValueError(f"No shape defined for {cls}")
            if "_order" not in data:
                data["_order"] = "C"
            _shape = data["_shape"]
            dshape = []  # find dynamic shapes
            for ii, d in enumerate(_shape):
                if d is None:
                    data["_is_static_shape"] = False
                    dshape.append(ii)
            if len(dshape) > 0:
                data["_is_static_shape"] = False
                data["_dshape_idx"] = dshape
                _data_offset += len(dshape) * 8  # space for dynamic shapes
                if len(_shape) > 1:
                    _data_offset += len(_shape) * 8  # space for strides
                else:
                    data["_strides"] = (static_size,)
            else:
                data["_is_static_shape"] = True
                data["_order"] = mk_order(data["_order"], _shape)
                data["_strides"] = get_strides(
                    _shape, data["_order"], static_size
                )

            if data["_is_static_shape"] and data["_is_static_type"]:
                _size = _itemtype._size
                for d in _shape:
                    _size *= d
                _size = _to_slot_size(_size)
            else:
                _size = None
                _data_offset += 8  # space for dynamic size

            data["_size"] = _size
            data["_data_offset"] = _data_offset
        # need to applied to derived classes as well
        if "_c_type" not in data:
            data["_c_type"] = name

        # determine has_refs
        if "_itemtype" in data.keys():
            if (
                hasattr(data["_itemtype"], "_has_refs")
                and data["_itemtype"]._has_refs
            ):
                data["_has_refs"] = True
            else:
                data["_has_refs"] = False

        return type.__new__(cls, name, bases, data)

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)

    def _get_offset(cls, index):
        return get_offset(index, cls._strides)

    def _get_n_items(cls):
        if cls._is_static_shape:
            return np.prod(cls._shape)
        else:
            raise ValueError("Cannot get n items from dynamic shapes")

    def __repr__(cls):
        return f"<array {cls.__name__}>"


class Array(metaclass=MetaArray):
    _shape: tuple
    _order: tuple
    _strides: tuple
    _itemtype: type
    _size: int
    _dshape: tuple
    _is_static_shape: bool
    _is_static_type: bool
    _data_offset: int

    @classmethod
    def mk_arrayclass(cls, itemtype, shape):
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
    def _inspect_args(cls, *args):
        """
        Determine:
        - size:
        - shape, order, strides
        - offsets
        - value: None if args contains dimensions else args[0]
        """
        # log.debug(f"get size for {cls} from {args}")
        info = Info()
        extra = {}
        if cls._size is not None:
            # static,static array
            if len(args) == 0:
                value = None
            elif len(args) == 1:
                (arg,) = args
                if arg is None:
                    value = None
                else:
                    shape = get_shape_from_array(arg, len(cls._shape))
                    if shape != cls._shape:
                        raise ValueError(f"shape not valid for {arg} ")
                    value = arg
            elif len(args) > 1:
                raise ValueError("too many arguments")
            size = cls._size
            shape = cls._shape
            order = cls._order
            strides = cls._strides
            offsets = np.empty(shape, dtype="int64")
            offset = 0
            for idx in iter_index(shape, order):
                offsets[idx] = offset
                offset += cls._itemtype._size
            assert size == _to_slot_size(offset)
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
                info.dshape = dshape
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
                info.offsets = offsets
                info.extra = extra

        info.shape = shape
        info.strides = strides
        info.size = size
        info.order = order
        info.value = value
        info.items = np.prod(shape)
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
        if not cls._is_static_shape:
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
        else:
            shape = cls._shape
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
        if hasattr(cls._itemtype, "_dtype") and hasattr(
            value, "dtype"
        ):  # is a scalar type:
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
        # determin resources
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

    def _get_size(self):
        if self.__class__._size is None:
            return Int64._from_buffer(self._buffer, self._offset)
        else:
            return self.__class__._size

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

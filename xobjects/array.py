import numpy as np

from .typeutils import get_a_buffer, Info
from .scalar import Int64

"""
array itemtype d1 d2 ...
array itemtype d1 : ...
array itemtype (d1,1) (d1,0) ...  F contiguos

There 6 kind of arrays from the combination of
    shape: static, dynamic
    item: static, dynamic


Data layout:
    - [size]: if not (static,static)
    - [d1 d2 ...]: dynamic dimensions (dynamic,*)
    - [offsets]: if itemtype is not static (*|dynamic)
    - data: array data

Array class:
    - _size
    - _shape: the shape in memory using C ordering
    - _order: the ordering of the index in the API
    - _itemtype
    - _is_static_shape
    - _is_static_type

Array instance:
    _dshape: index of dynamic shapes
    _shape: present if dynamic
"""


def get_shape_from_array(value):
    if hasattr(value, "shape"):
        return value.shape
    elif hasattr(value, "_shape"):
        return value._shape
    if hasattr(value, "__len__"):
        shape = (len(value),)
        if len(value) > 0:
            shape0 = get_shape_from_array(value[0])
            if shape0 == ():
                return shape
            for i in value[1:]:
                shapei = get_shape_from_array(i)
                if shapei != shape0:
                    raise ValueError(f"{value} not an array")
            return shape + shape0
        else:
            return shape
    else:
        return ()


def get_strides(shape, order):
    ss = 1
    strides = []
    for io in order:
        strides = [ss] + strides
        ss *= shape[io]
    return strides


def iter_index(shape, order):
    """return index in order of data layout"""
    for ii in np.ndindex(*shape):
        yield tuple([ii[oo] for oo in order])


def mk_order(order, shape):
    if order == "C":
        return list(range(len(shape)))
    elif order == "F":
        return list(range(len(shape) - 1, -1, -1))
    else:
        return order


def mk_getitem(itemtype, shape):
    is_static = not (itemtype._size is None or None in shape)
    if itemtype._size is None:
        itemsize = 8
    else:
        itemsize = itemtype._size
    # calculate strides
    ssv = itemsize
    sss = ()
    strides = [f"s{len(shape)-1}={ssv}"]
    for ii in range(len(shape) - 1, 0, -1):
        dd = shape[ii]
        if dd is None:
            sss += (f"d{ii}",)
        else:
            ssv *= dd
        strides.append(f'  s{ii-1}={ssv}*{"*".join(sss)}')

    indexes = ", ".join(f"i{ii}" for ii in range(len(shape)))
    out = [f"def __getitem__(self, {indexes}):"]
    if len(sss) > 0:
        out.append(f'  {",".join(sss)}=self._getshape()')
    out.extend(strides)
    offset = "+".join(f"i{ii}*s{ii}" for ii in range(len(shape)))
    out.append(f"  offset={offset}")
    if is_static is None:  # flexible size
        out.append(f"  base=self._offset+8+{len(sss)*8}")
    else:
        out.append(f"  base=self._offset")
    if itemtype._size is None:  # variable size
        out.append(f"  offset=Int64._from_buffer(self._buffer,base+offset)")
    out.append(f"  return self._itemtype._from_buffer(self._buffer,base+offset)")
    return "\n".join(out)


class MetaArray(type):
    def __new__(cls, name, bases, data):
        if "_itemtype" in data:  # specialized class
            _itemtype = data["_itemtype"]
            if _itemtype._size is None:
                data["_is_static_type"] = False
            else:
                data["_is_static_type"] = True
            if "_shape" not in data:
                raise ValueError("No shape defined for the Array")
            if "_order" not in data:
                data["_order"] = "C"
            _shape = data["_shape"]
            data["_is_static_shape"] = True
            for d in _shape:
                if d is None:
                    data["_is_static_shape"] = False

            if data["_is_static_shape"]:
                data["_order"] = mk_order(data["_order"], _shape)
            if data["_is_static_shape"] and data["_is_static_type"]:
                _size = _itemtype._size
                for d in _shape:
                    _size *= d
            else:
                _size = None
            data["_size"] = _size

        return type.__new__(cls, name, bases, data)


class Array(metaclass=MetaArray):
    @classmethod
    def mk_arrayclass(cls,itemtype,shape,order):
        pass

    @classmethod
    def _inspect_args(cls, *args):
        if cls._size is not None:
            # static,static array
            if len(args) == 1:
                shape = get_shape_from_array(args[0])
                if shape != cls._shape:
                    raise ValueError(f"shape not valid for {args[0]} ")
            elif len(args) > 1:
                raise ValueError(f"too many arguments")
            return Info(size=cls._size)
        else:
            info = Info()
            offset = 8  # space for size data
            if cls._is_static_shape:
                items = np.prod(cls._shape)
            else:
                # complete dimensions
                if not isinstance(args[0], int):  # init with array
                    value = np.array(args[0])
                    shape = value.shape
                    if hasattr(cls._shape):
                        dshape = []
                        for idim, ndim in enumerate(cls._shape):
                            if ndim is None:
                                dshape.append(idim)
                            else:
                                if shape[idem] != ndim:
                                    raise ValueError("Array: incompatible dimensions")
                    else:
                        dshape = shape
                else:
                    if hasattr(cls._shape):
                        shape = []
                        dshape = []  # index of dynamic shapes
                        for ndim in cls._shape:
                            if ndim is None:
                                shape.append(args[len(dshape)])
                                dshape.append(len(shape))
                            else:
                                shape.append(ndim)
                    else:
                        offset += 8  # space for ndim
                        shape = list(args)
                        dshape = shape
                # now we have shape, dshape
                info.shape = shape
                info.dshape = dshape
                info.order = mk_order(shape, cls._order)
                items = np.prod(shape)
            if cls._is_static_itemtype:
                offset += items * cls._itemtype  # starting of data
                info.data_offset = offset  # starting of data
                info.size=offset
            else:
                # args must be an array of correct dimensions
                extra = {}
                offsets = np.empty(shape,dtype='int64').transpose(cls._order)
                offset += items * 8
                for idx in iter_index(shape, order):
                    extra[idx]=cls._itemtype._inspect_args(value[idx])
                    offsets[idx]=offset
                    offset+=extra[idx].size
                info.extra = extra
                info.size=offset
            return info

    @classmethod
    def _from_buffer(cls, buffer, offset):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        coffset = offset
        if cls._size is None:
            self._size = Int64._from_buffer(self._buffer, coffset)
            coffset += 8
        if cls._shape is None:
            nd = Int64._from_buffer(self._buffer, offset)
            coffset += 8
            shape = []
            for ii in range(nd):
                shape.append(Int64._from_buffer(self._buffer, coffset))
                coffset += 8
            self._shape = shape
        elif not is_static_shape:
            shape = []
            for dd in cls._shape:
                if dd is None:
                    shape.append(Int64._from_buffer(self._buffer, coffset))
                    coffset += 8
                else:
                    shape.append(dd)
            self._shape = shape
        else:
            shape = cls._shape
        if not is_static_type:
            items = prod(shape)
            self._offsets = Int64._array_from_buffer(buffer, coffset, items)
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        if info is None:
            info = cls._inspect_args(value)
        header = []
        coffset = offset
        if cls._size is None:
            header.append(info.size)
        if not cls._is_static_shape:
            for ii, nd in enumerate(cls._shape):
                if nd is None:
                    header.append(info.shape[ii])
        if len(header) > 0:
            Int64.array_to_buffer(buffer, coffset, np.array(header, dtype="i8"))
            coffset += 8 * len(header)
        if not cls._is_static_type:
            Int64.array_to_buffer(buffer, coffset, info.offsets)
        if isinstance(value, np.ndarray) and hasattr(
            cls._itemtype, "_dtype"
        ):  # not robust try scalar classes
            if cls._itemtype._dtype == value.dtype:
                buffer.write(value.tobytes())
            else:
                buffer.write(value.astype(cls._itemtype._dtype).tobytes())
        else:
            for idx in iter_index(info.shape,cls._order):
                cls._itemtype._to_buffer(
                        buffer,
                        offset+info.offsets[idx],
                        value[idx],
                        info.extra.get(idx)
                        )

    def __init__(self, *args, _context=None, _buffer=None, _offset=None):
        # determin resources
        info = self.__class__._inspect_args(*args)

        self._buffer, self._offset = get_a_buffer(_context, _buffer, _offset)

        if info.value is not None:
           self.__class__._to_buffer(self._buffer, self._offset, info.value, info)

        if hasattr(info,'size'):
            self._size=info.size
        if hasattr(info,'shape'):
            self._shape=info.shape
        if hasattr(info,'offsets'):
            self._offsets=info.offsets

    def _get_size(self):
        if self.__class__._size is None:
            return Int64._from_buffer(self._buffer, self._offset)
        else:
            return self.__class__._size

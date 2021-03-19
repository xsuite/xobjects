import numpy as np

from .typeutils import get_a_buffer, Info
from .scalar import Int64

"""
array itemtype d1 d2 ...
array itemtype d1 : ...
array itemtype (d1,1) (d1,0) ...  F contiguos

There 6 kind of arrays from the combination of
    shape: static, nd, free
    item: static, dynamic


Data layout:
    - [size]: if not (static,static)
    - [ndim]: if free shape (free,*)
    - [d1 d2 ...]: dynamic dimensions (nd|free,*)
    - [offsets]: if itemtype is not static (*|dynamic)
    - data: array data

Array class:
    _size
    _shape
    _order
    _itemtype
    _is_static_shape
    _is_static_type

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

def get_strides(shape,order):
    ss=1
    strides=[]
    for io in order:
        strides=[ss]+strides
        ss*=shape[io]
    return strides

def iter_index(shape,order):
    idx=[0,0,0]
    strides=get_strides(shape,order)
    for io in order:
        pass



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
                data["_shape"] = None
            _shape = data["_shape"]
            if _shape is not None:
               _shape = data["_shape"]
               data["_is_static_shape"] = True
               for d in _shape:
                   if d is None:
                       data["_is_static_shape"] = False
            else:
                data["_is_static_shape"] = False
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
    def _inspect_args(cls, *args):
        if cls._size is not None:
            if len(args)==1:
               shape=get_shape_from_array(args[0])
               if shape!=cls._shape:
                   raise ValueError(f"shape not valid for {args[0]} ")
            elif len(args)>1:
                   raise ValueError(f"too many arguments")
            return Info(size=cls._size)
        else:
            info=Info()
            offset = 8  # space for size data
            if cls._is_static_shape:
                items = np.prod(cls._shape)
            else:
                # complete dimensions
                if not isinstance(args[0], int): # init with array
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
                info.shape=shape
                info.dshape=dshape
                items = np.prod(shape)
            if cls._is_static_itemtype:
                offset += items * cls._itemtype # starting of data
                info.data_offset=offset # starting of data
            else:
                # args must be an array
                offsets = []
                offset += items * 8
                for idx in iter_index(shape,order):
                    pass
                size += items * 8
            return info

    def __init__(self, *args, _context=None, _buffer=None, _offset=None):
        # determin resources
        size, offsets = self.__class__._inspect_args(*args)

        self._buffer, self._offset = get_a_buffer(_context, _buffer, _offset)

        # set structrure
        cls = self.__class__
        soffset = self._offset
        # set size
        if cls._size is None:
            Int64._to_buffer(self._buffer, soffset, size)
            soffset += 8
        # set dynamic dims
        if not cls._is_static_shape:
            for ii in self._dshape:
                Int64._to_buffer(self._buffer, soffset, size)
                soffset += 8
        # set dynamic dims offsets must be dtype(int64)
        if not cls._is_static_itemtype:
            self._buffer.write(soffset, offsets.data)
            soffset += len(offsets) * 8
        # set data
        if not isinstance(args[0], int):
            for idx in self._iter_dims():
                self[idx] = value[idx]

    @classmethod
    def _to_buffer(self, buffer, offset, data):
        ll = len(data)
        Int64._to_buffer(buffer, offset, ll)
        buffer.write(offset + 8, data)

    @classmethod
    def _from_buffer(self, buffer, offset):
        ll = Int64._from_buffer(buffer, offset)
        return buffer.read(buffer, offset + 8, ll).decode("utf8")

        if not cls._is_static_shape:
            if hasattr(self.__class__, "_shape"):
                tshape = self.__class__._shape
            else:
                tshape = [None] * len(shape)
            dshape = []
            for idim, ndim in enumerate(tshape):
                if ndim is None:
                    dshape.append(idim)
            self._dshape = dshape

    def _get_size(self):
        if self.__class__._size is None:
            return Int64._from_buffer(self._buffer, self._offset)
        else:
            return self.__class__._size

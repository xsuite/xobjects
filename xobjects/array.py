from .context import ByteArrayContext
from .scalar import Int64


"""
array mtype d1 d2 ...

layout:
[size] if not static
[d1 d2 ...] not fixed dimensions
[offsets] if mtype is not static
[data]

"""


def mk_getitem(mtype, dims):
    is_static = not (mtype._size is None or None in dims)
    if mtype._size is None:
        itemsize = 8
    else:
        itemsize = mtype._size
    # calculate strides
    ssv = itemsize
    sss = ()
    strides = [f"s{len(dims)-1}={ssv}"]
    for ii in range(len(dims) - 1, 0, -1):
        dd = dims[ii]
        if dd is None:
            sss += (f"d{ii}",)
        else:
            ssv *= dd
        strides.append(f'  s{ii-1}={ssv}*{"*".join(sss)}')

    indexes = ", ".join(f"i{ii}" for ii in range(len(dims)))
    out = [f"def __getitem__(self, {indexes}):"]
    if len(sss) > 0:
        out.append(f'  {",".join(sss)}=self._getdims()')
    out.extend(strides)
    offset = "+".join(f"i{ii}*s{ii}" for ii in range(len(dims)))
    out.append(f"  offset={offset}")
    if is_static is None:  # flexible size
        out.append(f"  base=self._offset+8+{len(sss)*8}")
    else:
        out.append(f"  base=self._offset")
    if mtype._size is None:  # variable size
        out.append(f"  offset=Int64._from_buffer(self._buffer,base+offset)")
    out.append(f"  return self._mtype._from_buffer(self._buffer,base+offset)")
    return "\n".join(out)


class MetaArray(type):
    def __new__(cls, name, bases, data):
        if "_mtype" and "_dims" in data:
            _mtype = data["_mtype"]
            _dims = data["_dims"]
            is_static = True
            if _mtype._size is None:
                is_static is False
            else:
                for d in _dims:
                    if d is None:
                        is_static = False
            if is_static:
                _size = _mtype._size
                for d in _dims:
                    _size *= d
            else:
                _size = None
            data["_size"] = _size

        return type.__new__(cls, name, bases, data)


class Array(metaclass=MetaArray):
    @classmethod
    def _to_buffer(self, buffer, offset, value):
        ll = len(data)
        Int64._to_buffer(buffer, offset, ll)
        buffer.write(offset + 8, data)

    @classmethod
    def _from_buffer(self, buffer, offset):
        ll = Int64._from_buffer(buffer, offset)
        return buffer.read(buffer, offset + 8, ll).decode("utf8")

    @classmethod
    def _get_size_from_args(cls, string_or_int):
        if isinstance(string_or_int, int):
            return string_or_int + 8
        elif isinstance(string_or_int, str):
            return len(string_or_int) + 8
        raise ValueError(
            f"String can accept only one integer or string and not `{string_or_int}`"
        )

    def _get_size(self):
        if self.__class__._size is None:
            return Int64._from_buffer(self._buffer, self._offset)
        else:
            return self.__class__._size

    def __init__(self, *dims, _buffer=None, _offset=None, _context=None):
        if _buffer is None:
            if _context is None:
                _context = ByteArrayContext()
            size = self.__class__._get_size_from_args(string_or_int)
            _buffer = _context.new_buffer(size)
            _offset = 0
        else:
            _offset = _buffer.allocate(size)
        self._buffer = _buffer
        self._offset = _offset

"""


TODO:
- String._to_buffer to use same context copy
- consider caching  the length
- consider using __slots__
- consider adding size in the class
"""

from .typeutils import get_a_buffer, Info, _to_slot_size
from .scalar import Int64

import logging

log = logging.getLogger(__name__)

class String:
    _size = None

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None):
        log.debug(f"{cls} to buffer {offset}  `{value}`")
        if info is None:  # string is always dynamic therefore index is necessary
            info = cls._inspect_args(value)
        size = info.size
        if isinstance(value, String):
            value = value.to_str()  # TODO not optimal
        if isinstance(value, str):
            data = bytes(value, "utf8")
            if size<len(data):
                raise ValueError(f"bug: mismatch `{value}` {size} {len(data)}")
            off=_to_slot_size(size)-len(data)
            data += b'\x00'*off
            stored_size = Int64._from_buffer(buffer, offset)
            if size > stored_size:
                raise ValueError(f"{value} to large to fit in {size}")
            buffer.write(offset + 8, data)
        else:
            raise ValueError("{value} not a string")

    @classmethod
    def _from_buffer(cls, buffer, offset):
        ll = Int64._from_buffer(buffer, offset)
        return buffer.read(offset + 8, ll).decode("utf8").rstrip("\x00")

    @classmethod
    def _inspect_args(cls, string_or_int):
        if isinstance(string_or_int, int):
            return Info(size=string_or_int + 8)
        elif isinstance(string_or_int, str):
            return Info(size=len(string_or_int) + 8)
        elif isinstance(string_or_int, cls):
            return Info(size=string_or_int._get_size())
        raise ValueError(
            f"String can accept only one integer or string and not `{string_or_int}`"
        )

    def _get_size(self):
        return Int64._from_buffer(self._buffer, self._offset)

    def __init__(self, string_or_int, _buffer=None, _offset=None, _context=None):
        new_object = False
        info = self.__class__._inspect_args(string_or_int)
        size = info.size
        self._buffer, self._offset = get_a_buffer(size, _context, _buffer, _offset)

        Int64._to_buffer(self._buffer, self._offset, size)

        if isinstance(string_or_int, str):
            self.set(string_or_int)

    def set(self, string):
        self.__class__._to_buffer(self._buffer, self._offset, string)

    def to_str(self):
        return self.__class__._from_buffer(self._buffer, self._offset)

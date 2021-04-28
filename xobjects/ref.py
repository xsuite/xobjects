import numpy as np
from .typeutils import Info

class MetaRef(type):

    def __getitem__(cls, rtype):
        return cls(rtype)

class Ref(metaclass=MetaRef):
    _size = 8

    def __init__(self, rtype):
        self._rtype = rtype

    def _from_buffer(self, buffer, offset):
        data = buffer.to_bytearray(offset, self._size)
        refoffset =  np.frombuffer(data, dtype=np.int64)[0]
        return self._rtype._from_buffer(buffer, refoffset)

    def _to_buffer(self, buffer, offset, value, info=None):

        if (isinstance(value, self._rtype)
                and value._buffer is buffer):
            refoffset = value._offset
        else:
            if np.isscalar(value):
                refoffset = int(value)
            else:
                raise NotImplementedError
        buffer.update_from_buffer(offset, np.int64(refoffset).tobytes())

    def __call__(self, value=None):
        if value is None:
            return -1
        else:
            raise NotImplementedError

    def _inspect_args(self, arg):
        return Info(size=self._size)

    #def __repr__(self):
    #    return self



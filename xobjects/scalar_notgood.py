"""
TODO:
    - make scalars as class instead of instances

"""

import logging

import numpy as np
from .context import Info

log = logging.getLogger(__name__)


class NumpyScalar:
    @classmethod
    def _from_buffer(self, buffer, offset):
        data = buffer.read(offset, self._size)
        log.debug(f"{self._size} {data}")
        return np.frombuffer(data, dtype=self._dtype)[0]

    @classmethod
    def _to_buffer(self, buffer, offset, value, info=None):
        data = self._dtype.type(value).tobytes()
        buffer.write(offset, data)

    @classmethod
    def _inspect_args(self, arg):
        return Info(size=self._size)

    @classmethod
    def _array_from_buffer(self, buffer, offset, count):
        return self.frombuffer(data, dtype=self._dtype, offset=offset, count=count)

    @classmethod
    def _array_to_buffer(self, buffer, offset, value):
        return buffer.write(offset, value.tobytes())


def gen_classes():
    types = {
        "Float": [16, 32, 64, 128],
        "Int": [8, 16, 32, 64],
        "UInt": [8, 16, 32, 64],
        "Complex": [64, 128, 256],
    }
    out = []
    for tt, sizes in types.items():
        for size in sizes:
            xtype = tt + str(size)
            dtype = tt.lower() + str(size)
            tmp = f"""class {xtype}(np.{dtype},NumpyScalar):
                _dtype = np.dtype('{dtype}')
                _size = {size}
"""
            out.append(tmp)
    return "\n".join(out)


exec(gen_classes())

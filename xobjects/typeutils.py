import numpy as np

from .context_cpu import ContextCpu

context_default = ContextCpu()


def get_a_buffer(size, context=None, buffer=None, offset=None):
    if buffer is None:
        if offset is not None:
            raise ValueError("Cannot set `offset` without buffer")
        if context is None:
            context = context_default
        buffer = context.new_buffer(size)
    if offset is None:
        offset = buffer.allocate(size)
    return buffer, offset


def dispatch_arg(f, arg):
    if isinstance(arg, tuple):
        return f(*arg)
    elif isinstance(arg, dict):
        return f(**arg)
    else:
        return f(arg)


class Info:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        args = [f"{k}={repr(v)}" for k, v in self.__dict__.items()]
        return f"Info({','.join(args)})"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _to_slot_size(size):
    "round to nearest multiple of 8"
    return (size + 7) & (-8)


def _is_dynamic(cls):
    return cls._size is None


def is_integer(i):
    return isinstance(i, (int, np.integer))


float2c = {2: "half", 4: "float", 8: "double", 16: "double[2]"}


default_conf = {
    "gpumem": "/*gpuglmem*/",
    "cpurestrict": "/*restrict*/",
    "inttype": "int64_t",
    "chartype": "char",
    "gpufun": "/*gpufun*/",
}


def get_c_type(typ):
    if hasattr(typ, "dtype"):
        ss = typ.dtype.str
        tt = ss[1]
        nb = int(ss[2:])
        if tt == "f":
            return float2c[nb]
        elif tt == "i":
            return f"int{nb*8}_t"
        elif tt == "u":
            return f"int{nb*8}_t"
        elif tt == "c":
            return f"{float2c[nb//2]}[2]"
        elif tt == "S":
            return f"char[{nb}]"
    elif hasattr(typ, "_c_type"):
        return typ._c_type
    else:
        raise ValueError(f"Cannot find C type for type {typ}")


class Register:
    def __init__(self):
        self.classes = {}

from .context import ContextDefault


def get_a_buffer(size, context=None, buffer=None, offset=None):
    if buffer is None:
        if offset is not None:
            raise ValueError("Cannot set `offset` without buffer")
        if context is None:
            context = ContextDefault()
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
    def __init__(self, **nargs):
        self.__dict__.update(nargs)

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

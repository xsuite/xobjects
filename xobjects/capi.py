from typing import NamedTuple, Bool

class field(NamedTuple):
    field_name: str
    type_name: str
    offset: int
    is_deferred: bool

class index(NamedTuple):
    shape: list(int)
    order: list(int)
    offset: int

def mk_getter(cname, spec):
    pass








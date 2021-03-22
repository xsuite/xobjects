from .context import ContextCpu, ContextCupy, ContextPyopencl, ContextDefault

from .scalar import (
    Float64,
    Float32,
    Int64,
    UInt64,
    Int32,
    UInt32,
    Int16,
    UInt16,
    Int8,
    UInt8,
)
from .array import Array
from .string import String
from .struct import Struct, Field
from .union import Union
from .unionref import UnionRef

# TODO: adapt the the tests and remove these
CLContext = ContextPyopencl
ByteArrayContext = ContextCpu

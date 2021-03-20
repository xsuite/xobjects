from .cpu import ContextCpu
from .py_opencl import ContextPyopencl
from .cu_py import ContextCupy

from .general import available

from .specialize_source  import specialize_source

ContextDefault = ContextCpu

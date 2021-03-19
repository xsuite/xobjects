from .cpu import ContextCpu
from .py_opencl import ContextPyopencl
from .cu_py import ContextCupy

from .general import available

ContextDefault = ContextCpu

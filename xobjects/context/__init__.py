from .cpu import ContextCpu
from .py_opencl import ContextPyopencl
from .cu_py import ContextCupy

ContextDefault = ContextCpu

ByteArrayContext = ContextCpu # For backward compatibility
CLContext = ContextPyopencl # For backward compatibility


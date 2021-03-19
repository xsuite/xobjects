from .py_opencl import ContextPyopencl
from .cpu import ContextCpu

ContextDefault = ContextCpu

ByteArrayContext = ContextCpu # For backward compatibility
CLContext = ContextPyopencl # For backward compatibility


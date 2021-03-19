from .py_opencl import CLContext
from .cpu import ContextCpu

ContextDefault = ContextCpu

ByteArrayContext = ContextCpu # For backward compatibility

from .py_opencl import CLContext
from .cpu import ContextCpu
from .general import Chunk, Info, get_a_buffer, dispatch_arg

ContextDefault = ContextCpu

ByteArrayContext = ContextCpu # For backward compatibility

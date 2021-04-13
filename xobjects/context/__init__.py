from .context_cpu import ContextCpu
from .context_pyopencl import ContextPyopencl
from .context_cupy import ContextCupy

from .general import available

from .specialize_source import specialize_source

context_default = ContextCpu()

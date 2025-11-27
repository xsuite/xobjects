from . import CPU
__all__ = ["CPU"]
try:
    from . import CUDA
    __all__.append("CUDA")
except ImportError:
    pass

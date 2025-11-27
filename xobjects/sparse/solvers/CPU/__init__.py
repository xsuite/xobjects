from scipy.sparse.linalg import splu as scipySuperLU
__all__ = ["scipySuperLU"]
try:
    from PyKLU import Klu as KLUSuperLU
    __all__.append("KLUSuperLU")
except ImportError:
    pass
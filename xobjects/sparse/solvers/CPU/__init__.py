from scipy.sparse.linalg import splu as scipySuperLU
from ....context import ModuleNotAvailableError

try:
    from PyKLU import Klu as KLUSuperLU
except (ModuleNotFoundError,ImportError) as e:
    def KLUSuperLU(*args, _import_err=e, **kwargs):
        raise ModuleNotAvailableError(
            "KLUSuperLU is not available. Could not import required backend."
            ) from _import_err

__all__ = ["scipySuperLU", "KLUSuperLU"]
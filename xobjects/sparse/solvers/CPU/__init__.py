from scipy.sparse.linalg import splu as scipysplu
try:
    from PyKLU import Klu as KLU
except (ModuleNotFoundError,ImportError) as e:
    def KLU(*args, _import_err=e, **kwargs):
        from ....context import ModuleNotAvailableError
        raise ModuleNotAvailableError(
            "KLU is not available. Could not import required backend."
            ) from _import_err

__all__ = ["scipysplu", "KLU"]
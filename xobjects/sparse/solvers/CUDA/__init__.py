from ....context import ModuleNotAvailableError
try:
    from ._cuDSSLU import DirectSolverSuperLU as cuDSSSuperLU
except (ModuleNotFoundError,ImportError) as e:
    def cuDSSSuperLU(*args, _import_err=e, **kwargs):
        raise ModuleNotAvailableError(
            "cuDSSSuperLU is not available. Could not import required backend."
            ) from _import_err
try:
    from ._luLU import luLU as CachedSuperLU
except (ModuleNotFoundError,ImportError) as e:
    def CachedSuperLU(*args, _import_err=e, **kwargs):
        raise ModuleNotAvailableError(
            "CachedSuperLU is not available. Could not import required backend."
            ) from _import_err
try:
    from cupyx.scipy.sparse.linalg import splu as CupySuperLU
except (ModuleNotFoundError,ImportError) as e:
    def CupySuperLU(*args, _import_err=e, **kwargs):
        raise ModuleNotAvailableError(
            "CupySuperLU is not available. Could not import required backend."
            ) from _import_err

__all__ = ["cuDSSSuperLU", "CachedSuperLU", "CupySuperLU"]
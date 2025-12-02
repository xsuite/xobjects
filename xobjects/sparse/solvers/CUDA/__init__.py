from ....context import ModuleNotAvailableError
try:
    from ._cuDSSLU import DirectSolverSuperLU as cuDSSSuperLU
except (ModuleNotFoundError,ImportError) as e:
    def cuDSSSuperLU(*args, **kwargs):
        raise ModuleNotAvailableError(
            "cuDSSSuperLU is not available. Could not import required backend."
            ) from e
try:
    from ._luLU import luLU as CachedSuperLU
except (ModuleNotFoundError,ImportError):
    def CachedSuperLU(*args, **kwargs):
        raise ModuleNotAvailableError(
            "CachedSuperLU is not available. Could not import required backend."
            ) from e
try:
    from cupyx.scipy.sparse.linalg import splu as CupySuperLU
except (ModuleNotFoundError,ImportError):
    def CupySuperLU(*args, **kwargs):
        raise ModuleNotAvailableError(
            "CupySuperLU is not available. Could not import required backend."
            ) from e

__all__ = ["cuDSSSuperLU", "CachedSuperLU", "CupySuperLU"]
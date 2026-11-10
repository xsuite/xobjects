try:
    from ._cuDSSLU import DirectSolverSuperLU as cuDSS
except (ModuleNotFoundError,ImportError) as e:
    def cuDSS(*args, _import_err=e, **kwargs):
        from ....context import ModuleNotAvailableError
        raise ModuleNotAvailableError(
            "cuDSS is not available. Could not import required backend."
            ) from _import_err
try:
    from ._luLU import luLU as cachedSpSM
except (ModuleNotFoundError,ImportError) as e:
    def cachedSpSM(*args, _import_err=e, **kwargs):
        from ....context import ModuleNotAvailableError
        raise ModuleNotAvailableError(
            "cachedSpSM is not available. Could not import required backend."
            ) from _import_err
try:
    from cupyx.scipy.sparse.linalg import splu as cupysplu
except (ModuleNotFoundError,ImportError) as e:
    def cupysplu(*args, _import_err=e, **kwargs):
        from ....context import ModuleNotAvailableError
        raise ModuleNotAvailableError(
            "cupysplu is not available. Could not import required backend."
            ) from _import_err

__all__ = ["cuDSS", "cachedSpSM", "cupysplu"]
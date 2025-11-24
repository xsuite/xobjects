import scipy.sparse
from numpy import ndarray as nparray
from typing import Optional, Literal, Union
from ..context import XContext
from ..context_cpu import ContextCpu
from ..context_cupy import ContextCupy
from ..context_pyopencl import ContextPyopencl
from .solvers._abstract_solver import SuperLUlikeSolver
try:
    from cupy import ndarray as cparray
    import cupyx.scipy.sparse
    from cupyx import cusparse
    _cupy_available = True
except:
    _cupy_available = False
    pass

def factorized_sparse_solver(A: Union[scipy.sparse.csr_matrix, 
                                      scipy.sparse.csc_matrix], 
                             n_batches: int = 0,
                             force_solver: Optional[
                                            Literal["scipySLU", 
                                                    "PyKLU",
                                                    "cuDSS", 
                                                    "CachedSLU", 
                                                    "cupySLU"]
                                                    ] = None,
                             solverKwargs: dict = None,
                             context: Union[
                                      Literal["cpu",
                                              "cupy",
                                              "pyopencl"],
                                      XContext
                                            ] = None
                            ) -> SuperLUlikeSolver:
    if solverKwargs is None:
        solverKwargs = {}
    if context is None:
        if isinstance(A, scipy.sparse.spmatrix) or isinstance(A, nparray):
            context = 'cpu'
        elif (_cupy_available and 
              (isinstance(A, cupyx.scipy.sparse.spmatrix) 
               or isinstance(A, cparray))):
            context = 'cupy'
        else:
            raise TypeError("Unsupported type for A")
    if context == "cpu" or isinstance(context, ContextCpu):
        assert isinstance(A, scipy.sparse.spmatrix), (
            "When using CPU context A must be a scipy.sparse matrix"
            )
        if 'permc_spec' not in solverKwargs:
            solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
        if force_solver is None or force_solver == "scipySLU":
            if A.shape[0]*n_batches < 10**5:
                import warnings
                warnings.warn("For small matrices, using PyKLU " 
                              "can provide improved performance")
            solver = scipy.sparse.linalg.splu(A.tocsc(),**solverKwargs)
        elif force_solver == "PyKLU":
            import PyKLU.klu as PyKLU
            solver = PyKLU.Klu(A.tocsc())
        else:
            raise ValueError("Unrecognized CPU Sparse solver. Available options: "
                             "scipySLU, PyKLU")
        
    elif context == "cupy" or isinstance(context, ContextCupy):
        if not _cupy_available:
            raise ModuleNotFoundError("No cupy module found. " \
                                      "ContextCupy unavailable")
        assert isinstance(A ,cupyx.scipy.sparse.csr_matrix), (
                "When using ContextCupy, input must be "
                "cupyx.scipy.sparse.csr_matrix")

        if force_solver is not None and force_solver != "cuDSS":
            if 'permc_spec' not in solverKwargs:
                solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
        if force_solver is None:
            import warnings
            try:
                from .solvers.CUDA._cuDSSLU import DirectSolverSuperLU
                solver = DirectSolverSuperLU(A, n_batches = n_batches, **solverKwargs)
            except (ModuleNotFoundError, RuntimeError) as e:
                warnings.warn("cuDSS not available. " 
                              "Falling back to Cached-SuperLU (spsm) "
                              f"Encountered Error: {e}")
                if 'permc_spec' not in solverKwargs:
                    solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
                try:
                    if cusparse.check_availability('csrsm2'):
                        raise RuntimeError("csrsm2 is avaiable. "
                                           "cupy SuperLU performs better "
                                           "than Cached-SuperLU (spsm)")
                    from .solvers.CUDA._luLU import luLU
                    solver = luLU(A, n_batches = n_batches, **solverKwargs)
                except RuntimeError as e:
                    warnings.warn("Cached-SuperLU (spsm) solver failed. " 
                                  "Falling back to cupy SuperLU. "
                                  f"Error encountered: {e}")
                    solver = cupyx.scipy.sparse.linalg.splu(A, **solverKwargs)
        elif force_solver == "cuDSS":
            from .solvers.CUDA._cuDSSLU import DirectSolverSuperLU
            solver = DirectSolverSuperLU(A, n_batches = n_batches, **solverKwargs)
        elif force_solver == "CachedSLU":
            from .solvers.CUDA._luLU import luLU
            solver = luLU(A, n_batches = n_batches, **solverKwargs)
        elif force_solver == "cupySLU":
            solver = cupyx.scipy.sparse.linalg.splu(A, **solverKwargs)
        else:
            raise ValueError("Unrecognized CUDA Sparse solver. Available options: "
                             "cuDSS, CachedSLU, cupySLU")
    elif context == "pyopencl" or isinstance(context, ContextPyopencl):
        raise NotImplementedError("No sparse solver is currently available "
                                  "for PyOpenCL context")
    else:
        raise ValueError("Invalid context. Available contexts are: " \
                         "cpu, cupy, pyopencl")
    return solver
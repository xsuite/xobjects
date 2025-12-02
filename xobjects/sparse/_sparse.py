import scipy.sparse
from numpy import ndarray as nparray
from typing import Optional, Literal, Union
from ..context import XContext
from ..context_cpu import ContextCpu
from ..context_cupy import ContextCupy
from ..context_pyopencl import ContextPyopencl
from .solvers._abstract_solver import SuperLUlikeSolver
from ..general import _print

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
                                            ] = None,
                             verbose: bool = False
                            ) -> SuperLUlikeSolver:
    """
    Build and return a factorized sparse linear solver on CPU or GPU.

    This function inspects the provided sparse matrix and execution context
    and returns a *factorized* solver object (SuperLU-like). The solver can
    then be reused to efficiently solve multiple linear systems with the same
    matrix `A`.

    The actual backend is chosen automatically based on:
      * The requested/derived `context` ("cpu", "cupy", "pyopencl"),
      * The availability of optional libraries (PyKLU, cuDSS, CuPy/CUSPARSE),
      * And the optional `force_solver` argument.

    On CPU:
      * Default: try PyKLU, fall back to `scipy.sparse.linalg.splu`.
      * You may force `"scipySLU"` or `"PyKLU"` explicitly.

    On CUDA/CuPy:
      * Default: try cuDSS (`DirectSolverSuperLU`), then fall back to:
          - `cupyx.scipy.sparse.linalg.splu` if CUSPARSE `csrsm2` is available, or
          - a cached SuperLU-based solver (`luLU`) and finally `cupyx.splu`.
      * You may force `"cuDSS"`, `"CachedSLU"`, or `"cupySLU"` explicitly.

    PyOpenCL:
      * Currently not supported and will raise `NotImplementedError`.

    Parameters
    ----------
    A : scipy.sparse.spmatrix or cupyx.scipy.sparse.spmatrix
        Sparse system matrix to factorize.

        The matrix is internally converted to the format expected by the
        chosen backend:

          * CPU context: converted to CSC (`scipy.sparse.csc_matrix`).
          * CuPy/GPU context: converted to CSR (`cupyx.scipy.sparse.csr_matrix`).

        For **best performance**, you should pass `A` already in the
        preferred format to avoid extra conversions:

          * If the context is (or will usually be) **CPU**, provide `A`
            as a CSC matrix (`A.tocsc()`).
          * If the context is **GPU/CuPy**, provide `A` as a CSR matrix
            (`A.tocsr()` or `cupyx.scipy.sparse.csr_matrix`).

        If `context` is `None`, it is still inferred from the type of `A`
        and the availability of CuPy, e.g.:

          * SciPy sparse or NumPy array → `"cpu"`,
          * CuPy sparse or CuPy array → `"cupy"`,
          * otherwise a `TypeError` is raised.

    n_batches : int, optional
        Controls the expected shape of the right-hand side (RHS) for GPU
        solvers and hence whether solves are treated as single or batched:

        * If ``n_batches == 0`` (default), the solver is configured for
          single-RHS solves and expects a vector RHS of shape ``(n,)``.
        * If ``n_batches > 0``, the solver is configured for batched solves
          and expects a 2D RHS array of shape ``(n, n_batches)`` (i.e.
          ``nrhs = n_batches``).

        This argument is primarily used by CUDA-based solvers (e.g. cuDSS and
        cached SuperLU) to preconfigure internal data structures for batched
        solves. It has no effect for CPU-based solvers.

    force_solver : {"scipySLU", "PyKLU", "cuDSS", "CachedSLU", "cupySLU"}, optional
        If provided, forces the use of a specific backend instead of the
        automatic selection:

        * `"scipySLU"` : Use `scipy.sparse.linalg.splu` (CPU).
        * `"PyKLU"`    : Use the `PyKLU.Klu` solver (CPU).
        * `"cuDSS"`    : Use CUDA/cuDSS-based `DirectSolverSuperLU` (GPU).
        * `"CachedSLU"`: Use CUDA cached SuperLU (`luLU`) (GPU).
        * `"cupySLU"`  : Use `cupyx.scipy.sparse.linalg.splu` (GPU).

        Using a solver that does not match the current `context` will result
        in a `ValueError`.

    solverKwargs : dict, optional
        Extra keyword arguments forwarded to the underlying solver constructor.
        If `None`, an empty dict is used.

        Some backends make use of `permc_spec` (matrix permutation strategy).
        When not explicitly provided and appropriate, this function sets
        `permc_spec="MMD_AT_PLUS_A"` as a sensible default for the matrices that 
        will typically be encountered in an xobjects workflow.

    context : {"cpu", "cupy", "pyopencl"} or XContext, optional
        Execution context. Can be either:

        * A string:
            - `"cpu"`: Use CPU-based solvers (SciPy / PyKLU).
            - `"cupy"`: Use CuPy/CUDA-based solvers.
            - `"pyopencl"`: PyOpenCL context (currently unsupported).
        * A context object instance:
            - `ContextCpu`
            - `ContextCupy`
            - `ContextPyopencl`

        If `None`, the context is inferred from `A` as described above.

    verbose : bool, optional
        If `True`, prints debug messages describing the solver-selection
        process, fallbacks, and the final solver that is returned.

    Returns
    -------
    SuperLUlikeSolver
        A factorized solver object compatible with SciPy’s `splu`-like
        interface (i.e. typically exposing a `solve` method and related
        accessors). The exact concrete type depends on the backend:
          * CPU:
              - `scipy.sparse.linalg.SuperLU` (for `"scipySLU"`),
              - `PyKLU.Klu` (for `"PyKLU"`).
          * CUDA/CuPy:
              - `DirectSolverSuperLU` (cuDSS),
              - `luLU` (cached SuperLU),
              - `cupyx.scipy.sparse.linalg.SuperLU` (for `"cupySLU"`).

    Raises
    ------
    TypeError
        If the type of `A` is unsupported when inferring the context.

    AssertionError
        If `A` does not match the required type for the chosen context

    ModuleNotFoundError
        If a requested solver backend depends on a module that is not
        installed (e.g. CuPy, PyKLU, cuDSS), and no fallback is available.

    RuntimeError
        If a requested GPU solver fails during initialization.

    NotImplementedError
        If `context` is `"pyopencl"` or `ContextPyopencl`, since no sparse
        solver is currently implemented for that backend.

    ValueError
        If an invalid `context` string is provided, or if `force_solver`
        does not match any known solver for the active context.

    Notes
    -----
    - For best performance on CPU, PyKLU is preferred when available.
    - For CuPy/CUDA, cuDSS is preferred when available, and this function
      will automatically fall back to other solvers if cuDSS is not present
      or fails at runtime.
    - The returned solver is *factorized* and should be reused to solve
      multiple right-hand sides efficiently.

    Examples
    --------
    Factorize a SciPy sparse matrix on CPU with automatic solver selection:

    >>> A = scipy.sparse.random(1000, 1000, density=0.01, format="csr")
    >>> solver = factorized_sparse_solver(A)
    >>> x = solver.solve(b)

    Explicitly request the SciPy SuperLU solver on CPU:

    >>> solver = factorized_sparse_solver(A, force_solver="scipySLU")

    Using CuPy and cuDSS (requires CuPy and cuDSS bindings):

    >>> A_gpu = cupyx.scipy.sparse.csr_matrix(A)
    >>> solver = factorized_sparse_solver(A_gpu, context="cupy")
    >>> x_gpu = solver.solve(b_gpu)
    """
    if solverKwargs is None:
        solverKwargs = {}
    if context is None:
        dbugprint(verbose, "No context provided. " \
                           "Context will be inferred from matrix")
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
        A = A.tocsc() # CPU Solvers require csc format
        if 'permc_spec' not in solverKwargs:
            solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
        if force_solver is None:
            dbugprint(verbose, "No solver requested. " \
                               "Picking best solver for CPU Context")
            try:
                dbugprint(verbose, "Attempting to use PyKLU")
                import PyKLU
                solver = PyKLU.Klu(A)
                dbugprint(verbose, "PyKLU succeeded")
            except (ModuleNotFoundError, RuntimeError) as e:
                dbugprint(verbose, "PyKLU failed. " \
                                   "Falling back to scipy.splu \n"
                                  f"Encountered error: {e}")
                
                solver = scipy.sparse.linalg.splu(A,**solverKwargs)
        elif force_solver == "scipySLU":
            solver = scipy.sparse.linalg.splu(A,**solverKwargs)
        elif force_solver == "PyKLU":
            import PyKLU
            solver = PyKLU.Klu(A)
        else:
            raise ValueError("Unrecognized CPU Sparse solver. Available options: "
                             "scipySLU, PyKLU")
        
    elif context == "cupy" or isinstance(context, ContextCupy):
        if not _cupy_available:
            raise ModuleNotFoundError("No cupy module found. " \
                                      "ContextCupy unavailable")
        assert isinstance(A ,cupyx.scipy.sparse.spmatrix), (
                "When using ContextCupy, input must be "
                "cupyx.scipy.sparse matrix")
        
        A = A.tocsr() # GPU solvers require csr format
        if force_solver is not None and force_solver != "cuDSS":
            if 'permc_spec' not in solverKwargs:
                solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
        if force_solver is None:
            dbugprint(verbose, "No solver requested. " \
                               "Picking best solver for Cupy Context")
            import warnings
            try:
                dbugprint(verbose, "Attempting to use cuDSS Solver")
                from .solvers.CUDA._cuDSSLU import DirectSolverSuperLU
                solver = DirectSolverSuperLU(A, n_batches = n_batches, **solverKwargs)
                dbugprint(verbose, "cuDSS succeeded")
            except (ModuleNotFoundError, RuntimeError) as e:
                dbugprint(verbose, "cuDSS failed. \n"
                                    f"Encountered Error: {e}")
                warnings.warn("cuDSS not available. Performance will be degraded")                               
                if 'permc_spec' not in solverKwargs:
                    solverKwargs = solverKwargs | {"permc_spec":"MMD_AT_PLUS_A"}
                if cusparse.check_availability('csrsm2'):
                    dbugprint(verbose, "csrsm2 available. Using cupyx.splu solver")
                    solver = cupyx.scipy.sparse.linalg.splu(A, **solverKwargs)
                else:
                    try:
                        dbugprint(verbose, "csrms2 unavailable. " \
                                           "Attempting to use CachedSuperLU (spsm)")
                        from .solvers.CUDA._luLU import luLU
                        solver = luLU(A, n_batches = n_batches, **solverKwargs)
                        dbugprint(verbose, "CachedSuperLU succeeded")
                    except RuntimeError as e:
                        dbugprint(verbose, "CachedSuperLU failed. \n"
                                          f"Encountered error: {e} \n"
                                           "Falling back to cupyx.splu with spsm")
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
    dbugprint(verbose, "Returning solver: " + str(solver))
    return solver

def dbugprint(verbose: bool, text: str):
    if verbose:
        _print("[xo.sparse] "+text)
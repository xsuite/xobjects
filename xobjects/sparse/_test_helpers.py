import numpy.linalg as npl
import scipy.sparse.linalg as scspl
from scipy.sparse import issparse

def issymmetric(A, tol=0):
    if A.shape[0] != A.shape[1]:
        return False
    diff = A - A.T
    if tol == 0:
        return diff.nnz == 0
    else:
        # tolerance-based check
        return abs(diff).max() <= tol

def rel_residual(A,x,b):
    if hasattr(A, "get"):
        A = A.get()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(b, "get"):
        b = b.get()
    assert issparse(A), ("A must be a sparse matrix")

    return npl.norm(A@x - b) / (npl.norm(b))

def assert_residual_ok(res_ref, res_solver,
                       abs_tol=1e-12,
                       factor=10):
    """
    Check that our solver's residual is both:
      - absolutely small enough (abs_tol),
      - not catastrophically worse than the reference (factor * res_ref).
    """
    # sanity: reference solver itself should be good
    assert res_ref < abs_tol, f"Reference residual too large: {res_ref}"

    # absolute bound
    assert res_solver < abs_tol, (
        f"Residual {res_solver} exceeds absolute tolerance {abs_tol}"
    )

    # relative bound vs reference
    assert res_solver <= factor * res_ref, (
        f"Residual {res_solver} not within factor {factor} of "
        f"reference residual {res_ref}"
    )

# ---- Unused rn, but could be useful for tests ----

def assert_close_to_precision(value, precision):
    assert value <= precision, (f"Value {value} not within precision {precision}")

def assert_residual_close(reference_residual, residual, tolerance = 10):
    assert residual <= tolerance*reference_residual, (
        f"Residual {residual} not within tolerance "
        f"O({tolerance}) of reference residual {reference_residual}")
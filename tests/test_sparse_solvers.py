import numpy as np
import scipy.sparse as sp
import xobjects as xo
from xobjects.test_helpers import fix_random_seed
import pytest

PARAMS = [
    ("scipySLU",  xo.ContextCpu()),
    ("PyKLU",     xo.ContextCpu()),
    ("cuDSS",     xo.ContextCupy()),
    ("CachedSLU", xo.ContextCupy()),
    ("cupySLU",   xo.ContextCupy()),
]

REL_TOL = 1e-4
ABS_TOL = 1e-8
SPARSE_SYSTEM_SIZE = 5000 # (n,n) matrix
NUM_BATCHES = 10

def batch_vectors_as_matrix(vector_list):
    return np.asfortranarray(np.moveaxis(np.array(vector_list),0,-1))

@fix_random_seed(1337)
def make_random_sparse_system(n, nbatches, density=0.01):
    A = sp.random(
        n, n,
        density=density,
        format="csc",
        random_state=np.random,
        data_rvs=np.random.standard_normal
    )
    # Make it nonsingular & better conditioned:
    # Add something on the diagonal so pivots aren't tiny/zero
    A = A + sp.eye(n, format="csc") * 5.0  # tweak factor as you like
    b_array = []
    if nbatches == 0:
        b = np.random.standard_normal(n)
        b_array.append(b)
    else:
        for i in range(nbatches):
            b = np.cos(2*i/(nbatches-1)*np.pi) * np.random.standard_normal(n)
            b_array.append(b)
        b = batch_vectors_as_matrix(b_array)
    solver = sp.linalg.splu(A)
    x = solver.solve(b)
    return (A, b, x, b_array)

@fix_random_seed(1337)
def make_tridiagonal_system(n, nbatches):
    main = 2.0 + np.abs(np.random.standard_normal(n))
    lower = np.random.standard_normal(n-1)
    upper = np.random.standard_normal(n-1)
    A = sp.diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        format="csc"
    )
    b_array = []
    if nbatches == 0:
        b = np.random.standard_normal(n)
        b_array.append(b)
    else:
        for i in range(nbatches):
            b = np.cos(2*i/(nbatches-1)*np.pi) * np.random.standard_normal(n)
            b_array.append(b)
        b = batch_vectors_as_matrix(b_array)
    solver = sp.linalg.splu(A)
    x = solver.solve(b)
    return (A, b, x, b_array)

random_system = make_random_sparse_system(SPARSE_SYSTEM_SIZE, 0)
tridiag_system = make_tridiagonal_system(SPARSE_SYSTEM_SIZE, 0)

@pytest.mark.parametrize("test_solver,test_context", PARAMS)
@pytest.mark.parametrize("sparse_system", [random_system, tridiag_system])
def test_vector_solve(test_solver, test_context, sparse_system):
    A_sp, b_sp, x_sp, _ = sparse_system
    np.testing.assert_allclose(b_sp, A_sp@x_sp, 
                               rtol = REL_TOL, atol = ABS_TOL) #Verify Scipy result
    if "Cpu" in str(test_context):
        A = test_context.splike_lib.sparse.csc_matrix(A_sp)
    if "Cupy" in str(test_context):
        A = test_context.splike_lib.sparse.csr_matrix(A_sp)
    b_test = test_context.nparray_to_context_array(b_sp)
    b = test_context.nplike_lib.asfortranarray(b_test)
    solver = xo.sparse.factorized_sparse_solver(A, 
                                                force_solver = test_solver, 
                                                context = test_context
                                                )
    x = solver.solve(b)
    
    xo.assert_allclose(x_sp, x, rtol = REL_TOL, atol = ABS_TOL)
    xo.assert_allclose(b, A@x, rtol = REL_TOL, atol = ABS_TOL)

random_system = make_random_sparse_system(SPARSE_SYSTEM_SIZE, NUM_BATCHES)
tridiag_system = make_tridiagonal_system(SPARSE_SYSTEM_SIZE, NUM_BATCHES)

@pytest.mark.parametrize("test_solver,test_context", PARAMS)
@pytest.mark.parametrize("sparse_system", [random_system, tridiag_system])
def test_batched_solve(test_solver, test_context, sparse_system):
    A_sp, b_sp, x_sp, _ = sparse_system
    np.testing.assert_allclose(b_sp, A_sp@x_sp, 
                               rtol = REL_TOL, atol = ABS_TOL) #Verify Scipy result
    if "Cpu" in str(test_context):
        A = test_context.splike_lib.sparse.csc_matrix(A_sp)
    if "Cupy" in str(test_context):
        A = test_context.splike_lib.sparse.csr_matrix(A_sp)
    b_test = test_context.nparray_to_context_array(b_sp)
    b = test_context.nplike_lib.asfortranarray(b_test)
    solver = xo.sparse.factorized_sparse_solver(A, 
                                                n_batches = NUM_BATCHES,
                                                force_solver = test_solver, 
                                                context = test_context
                                                )
    x = solver.solve(b)
    
    xo.assert_allclose(x_sp, x, rtol = REL_TOL, atol = ABS_TOL)
    xo.assert_allclose(b, A@x, rtol = REL_TOL, atol = ABS_TOL)
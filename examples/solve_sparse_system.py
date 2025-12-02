# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import scipy.sparse as sp
import numpy as np
from xobjects.sparse._test_helpers import rel_residual

'''
The goal of this example is to provide a short user guide for the xo.sparse module.

The xo.sparse module can be used to solve sparse linear systems of equations:
                                    A*x = b
where A is a sparse matrix.

Currently this module only supports CPU and Cupy contexts. This module contains a 
variety of solvers for different contexts, with consistent APIs. The intended use
is to reuse the same LHS for many solves, so the solvers work as follows:

solver(A) # Performs decomposition/factorization
solver.solve(b) # Solves Ax = b using precomputed factors

For optimal performance across backends b should be a column-major (F Contiguous)
array or vector.

The intended interface for this module is:

xo.sparse.factorized_sparse_solver()

The function includes detailed documentation for usage, but in short, it returns
the best performing solver based on the context and available modules. If the 
context is not explicitly defined, it is inferred based on the input matrix.

This is how modules that build upon this functionality within xsuite should interact
with the xo.sparse module, so that cross-platform compatibility is guaranteed.

For development and convenience purposes xo.sparse provides the:
xo.sparse.solvers module

which provides the following:
xo.sparse.solvers.CPU.
    - scipysplu : Alias for scipy SuperLU 
    - KLUSuperLU : Alias for PyKLU
xo.sparse.solvers.CPU.
    - cuDSS : nvmath.sparse.advanced.DirectSolver Wrapper with a SuperLU-like interface
    - cachedSpSM : Rewrite of cupy's SuperLU to cache the SpSM analysis step
                   offering massive speedups compared to cupy splu when the only
                   available backend is SpSM
    - cupysplu : Alias for scipy SuperLU 
'''

# Example: solve small matrix system
n = 5
# Create matrix
main = 2.0 + np.abs(np.random.standard_normal(n))
lower = np.random.standard_normal(n-1)
upper = np.random.standard_normal(n-1)
A = sp.diags(
    diagonals=[lower, main, upper],
    offsets=[-1, 0, 1],
    format="csc"
)

# Create solver:
print("Solver selection process: ")
solver = xo.sparse.factorized_sparse_solver(A, verbose = True)

# Generate random vector to solve:
b = np.random.standard_normal(n)

# Solve system
x = solver.solve(b)

# Calculate relative residual to assess solver:
res = rel_residual(A,x,b)
print("Relative residual of solution ", res)
print("Residual should be small (<10^-12)")

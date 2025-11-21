import cupy as _cupy
import numpy as _numpy
import cupyx.scipy.sparse
from cupyx.scipy.sparse.linalg import SuperLU
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupyx.cusparse import _dtype_to_IndexType, SpMatDescriptor, DnMatDescriptor, check_availability
import scipy.sparse.linalg
try:
    import scipy.sparse.linalg
    scipy_available = True
except ImportError:
    scipy_available = False
from line_profiler import profile

class CachedAbSolver:
    
    def __init__(self, a, b, alpha=1.0, lower=True, unit_diag=False, transa=False):
        if not check_availability('spsm'):
            raise RuntimeError('spsm is not available.')

        # Canonicalise transa
        if transa is False:
            transa = 'N'
        elif transa is True:
            transa = 'T'
        elif transa not in 'NTH':
            raise ValueError(f'Unknown transa (actual: {transa})')

        # Check A's type and sparse format
        if cupyx.scipy.sparse.isspmatrix_csr(a):
            pass
        elif cupyx.scipy.sparse.isspmatrix_csc(a):
            if transa == 'N':
                a = a.T
                transa = 'T'
            elif transa == 'T':
                a = a.T
                transa = 'N'
            elif transa == 'H':
                a = a.conj().T
                transa = 'N'
            lower = not lower
        elif cupyx.scipy.sparse.isspmatrix_coo(a):
            pass
        else:
            raise ValueError('a must be CSR, CSC or COO sparse matrix')
        assert a.has_canonical_format

        # Check B's ndim
        if b.ndim == 1:
            is_b_vector = True
            b = b.reshape(-1, 1)
        elif b.ndim == 2:
            is_b_vector = False
        else:
            raise ValueError('b.ndim must be 1 or 2')
        self.is_b_vector = is_b_vector

        # Check shapes
        if not (a.shape[0] == a.shape[1] == b.shape[0]):
            raise ValueError('mismatched shape')

        # Check dtypes
        dtype = a.dtype
        if dtype.char not in 'fdFD':
            raise TypeError('Invalid dtype (actual: {})'.format(dtype))
        if dtype != b.dtype:
            raise TypeError('dtype mismatch')
        self.dtype = dtype

        # Prepare fill mode
        if lower is True:
            fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER
        elif lower is False:
            fill_mode = _cusparse.CUSPARSE_FILL_MODE_UPPER
        else:
            raise ValueError('Unknown lower (actual: {})'.format(lower))
        self.fill_mode = fill_mode

        # Prepare diag type
        if unit_diag is False:
            diag_type = _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT
        elif unit_diag is True:
            diag_type = _cusparse.CUSPARSE_DIAG_TYPE_UNIT
        else:
            raise ValueError('Unknown unit_diag (actual: {})'.format(unit_diag))
        self.diag_type = diag_type
        self.transa = transa
        # Prepare op_a
        if transa == 'N':
            op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        elif transa == 'T':
            op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:  # transa == 'H'
            if dtype.char in 'fd':
                op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
            else:
                op_a = _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
        self.op_a = op_a
        # Prepare op_b
        self.op_b = self._get_opb(b)
        
        # Allocate space for matrix C. Note that it is known cusparseSpSM requires
        # the output matrix zero initialized.
        m, _ = a.shape
        if self.op_b == _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE:
            _, n = b.shape
        else:
            n, _ = b.shape
        c_shape = m, n
        self.c_shape = c_shape

        self._perform_analysis(a, b, alpha=alpha)

    def _perform_analysis(self, a, b, alpha=1.0):
        """Solves a sparse triangular linear system op(a) * x = alpha * op(b).

        Args:
            a (cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.coo_matrix):
                Sparse matrix with dimension ``(M, M)``.
            b (cupy.ndarray): Dense matrix with dimension ``(M, K)``.
            alpha (float or complex): Coefficient.
            lower (bool):
                True: ``a`` is lower triangle matrix.
                False: ``a`` is upper triangle matrix.
            unit_diag (bool):
                True: diagonal part of ``a`` has unit elements.
                False: diagonal part of ``a`` has non-unit elements.
            transa (bool or str): True, False, 'N', 'T' or 'H'.
                'N' or False: op(a) == ``a``.
                'T' or True: op(a) == ``a.T``.
                'H': op(a) == ``a.conj().T``.
        """
        
        c = _cupy.zeros(self.c_shape, dtype=a.dtype, order='f')

        # Prepare descriptors and other parameters
        self.handle = _device.get_cusparse_handle()
        self.mat_a = SpMatDescriptor.create(a)
        mat_b = DnMatDescriptor.create(b)
        mat_c = DnMatDescriptor.create(c)
        self.spsm_descr = _cusparse.spSM_createDescr()
        self.alpha = _numpy.array(alpha, dtype=c.dtype).ctypes
        self.cuda_dtype = _dtype.to_cuda_dtype(c.dtype)
        self.algo = _cusparse.CUSPARSE_SPSM_ALG_DEFAULT

        try:
            # Specify Lower|Upper fill mode
            self.mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_FILL_MODE, self.fill_mode)

            # Specify Unit|Non-Unit diagonal type
            self.mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_DIAG_TYPE, self.diag_type)

            # Allocate the workspace needed by the succeeding phases
            buff_size = _cusparse.spSM_bufferSize(
                self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
                mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr)
            self.buff = _cupy.empty(buff_size, dtype=_cupy.int8)

            # Perform the analysis phase
            _cusparse.spSM_analysis(
                self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
                mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr, self.buff.data.ptr)
        except Exception as e:
            raise RuntimeError('spSM_analysis failed.') from e
    
    def _get_opb(self, b):
        # Prepare op_b
        if b._f_contiguous:
            op_b = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        elif b._c_contiguous:
            if _cusparse.get_build_version() < 11701:  # earlier than CUDA 11.6
                raise ValueError('b must be F-contiguous.')
            b = b.T
            op_b = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:
            raise ValueError('b must be F-contiguous or C-contiguous.')
        return op_b
    @profile
    def solve(self, b):
        assert b.dtype == self.dtype
        assert self._get_opb(b) == self.op_b
        
        if b.ndim == 1:
            is_b_vector = True
            b = b.reshape(-1, 1)
        elif b.ndim == 2:
            is_b_vector = False
        else:
            raise ValueError('b.ndim must be 1 or 2')
        self.is_b_vector = is_b_vector
        
        c = _cupy.zeros(self.c_shape, dtype=self.dtype, order='f')
        mat_b = DnMatDescriptor.create(b)
        mat_c = DnMatDescriptor.create(c)
        try:
            # Executes the solve phase
            _cusparse.spSM_solve(
                self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
                mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr, self.buff.data.ptr)
        finally:
            _cupy.cuda.get_current_stream().synchronize()
        # Reshape back if B was a vector
        if self.is_b_vector:
            c = c.reshape(-1)
        return c

    # def __del__(self):
    #     # Destroy matrix descriptor
    #     print("Deleting solver obj...")
    #     _cusparse.spSM_destroyDescr(self.spsm_descr)

class luLU(SuperLU):

    def __init__(self, A, 
                 trans = 'N', 
                 permc_spec=None, 
                 n_batches: int = 0, 
                 diag_pivot_thresh=None, 
                 relax=None,
                 panel_size=None, 
                 options={}):
        if not check_availability('spsm'):
            raise RuntimeError('spsm is not available.')
        if not scipy_available:
            raise RuntimeError('scipy is not available')
        if not cupyx.scipy.sparse.isspmatrix(A):
            raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
        if A.shape[0] != A.shape[1]:
            raise ValueError('A must be a square matrix (A.shape: {})'
                            .format(A.shape))
        if A.dtype.char not in 'fdFD':
            raise TypeError('Invalid dtype (actual: {})'.format(A.dtype))

        a = A.get().tocsc()
        a_slu = scipy.sparse.linalg.splu(
            a, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh,
            relax=relax, panel_size=panel_size, options=options)
        super().__init__(a_slu)
        self.b_dtype = self.L.dtype
        self._init_solvers(trans=trans, n_batches = n_batches)
    
    def _init_solvers(self,trans = 'N', n_batches = 0):
        if n_batches == 0:
            b_sample = _cupy.zeros(self.shape[0], 
                                   dtype=self.b_dtype)
        else:
            b_sample = _cupy.zeros((self.shape[0],n_batches), 
                                   dtype=self.b_dtype, order = 'F')
        self.trans = trans
        self.b_shape = b_sample.shape
        self.Lsolver = CachedAbSolver(self.L, b_sample, lower=True, transa=self.trans)
        self.Usolver = CachedAbSolver(self.U, b_sample, lower=False, transa=self.trans)
        # self.Usolver = CachedAbSolver(self.U.T, b_sample, lower=True, transa="T") #Can improve performance at times
    @profile
    def solve(self, rhs, trans='N'):
        """Solves linear system of equations with one or several right-hand sides.

        Args:
            rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
                ``(M)`` or ``(M, K)``.
            trans (str): 'N', 'T' or 'H'.
                'N': Solves ``A * x = rhs``.
                'T': Solves ``A.T * x = rhs``.
                'H': Solves ``A.conj().T * x = rhs``.

        Returns:
            cupy.ndarray:
                Solution vector(s)
        """  # NOQA
        from cupyx import cusparse
        if trans != self.trans:
            raise AssertionError("Solve function assumes cached configuration. " \
            "Rebuild cache by calling _init_solvers with desired configuration.")
        if not isinstance(rhs, _cupy.ndarray):
            raise TypeError('ojb must be cupy.ndarray')
        if rhs.ndim not in (1, 2):
            raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.
                             format(rhs.ndim))
        if rhs.shape[0] != self.shape[0]:
            raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'
                             .format(self.shape, rhs.shape))
        assert rhs.shape == self.b_shape, (
            "Cached solver can only accept RHS with the same shape "
            f"as the initialized value. {self.b_shape}. "
            "The initialized RHS shape can be changed by initializing "
            "using a different value for n_batches"
        )
        if trans not in ('N', 'T', 'H'):
            raise ValueError('trans must be \'N\', \'T\', or \'H\'')

        x = rhs.astype(self.L.dtype)
        if trans == 'N':
            if self.perm_r is not None:
                if x.ndim == 2 and x._f_contiguous:
                    x = x.T[:, self._perm_r_rev].T  # want to keep f-order
                else:
                    x = x[self._perm_r_rev]
            x = self.Lsolver.solve(x)
            x = self.Usolver.solve(x)
            if self.perm_c is not None:
                x = x[self.perm_c]
        else:
            if self.perm_c is not None:
                if x.ndim == 2 and x._f_contiguous:
                    x = x.T[:, self._perm_c_rev].T  # want to keep f-order
                else:
                    x = x[self._perm_c_rev]
            x = self.Usolver.solve(x)
            x = self.Lsolver.solve(x)
            if self.perm_r is not None:
                x = x[self.perm_r]

        if not x._f_contiguous:
            # For compatibility with SciPy
            x = x.copy(order='F')
        return x
    
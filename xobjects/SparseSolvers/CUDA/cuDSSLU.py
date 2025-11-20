import cupy as cp
import cupyx.scipy.sparse as sp
import nvmath

class DirectSolverSuperLU(nvmath.sparse.advanced.DirectSolver):
    """cuDSS-based direct solver; reuse factors for many RHS (SuperLU-style)."""

    def __init__(self, A, *, 
                 n_batches: int = 0,
                 assume_general=True, 
                 order_rhs='auto', 
                 **kwargs):
        # 1) Validate A
        if not isinstance(A, sp.csr_matrix):
            raise TypeError("A must be cupyx.scipy.sparse.csr_matrix")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if A.indices.dtype != cp.int32 or A.indptr.dtype != cp.int32:
            A = sp.csr_matrix((A.data, A.indices.astype(cp.int32), A.indptr.astype(cp.int32)),
                              shape=A.shape)
        # 2) Sample RHS to initialize parent
        if n_batches == 0:
            b_sample = cp.zeros(A.shape[0], dtype=A.dtype)
        else:
            b_sample = cp.zeros((A.shape[0],n_batches), dtype=A.dtype, order = 'F')
        self.b_dtype = A.dtype
        self.b_shape = b_sample.shape
        super().__init__(A, b_sample, **kwargs)

        # 3) Optional: configure plan
        pc = self.plan_config
        if assume_general:
            # LU path (cuDSS will often infer this, but we can be explicit if available)
            try:
                pc.matrix_type = nvmath.sparse.advanced.DirectSolverMatrixType.GENERAL
            except AttributeError:
                pass  # older wheels infer type; safe to ignore
        # Expose a simple RHS layout policy
        self._order_rhs = order_rhs  # 'F'|'C'|'auto'

        # 4) Build & factorize once; warmup solve to build internal caches
        super().plan()
        self.fac_info = super().factorize()
        # Warmup on zeros (uses parent solve())
        super().reset_operands(b=b_sample)
        super().solve()

    def _prepare_rhs(self, b):
        # Accept 1-D or 2-D; ensure dtype/device and layout
        if b.dtype != self.b_dtype:
            raise TypeError(f"RHS dtype {b.dtype} does not match matrix dtype {self.b_dtype}")
        if b.ndim == 1:
            return b  # vector is fine
        if self._order_rhs == 'auto':
            # cuDSS prefers column-major for (n,k)
            return cp.array(b, order='F', copy=False)
        return cp.array(b, order=self._order_rhs, copy=False)

    def solve(self, b):
        """Solve A x = b using cached factors. Accepts (n,) or (n,k) RHS."""
        assert b.shape == self.b_shape, (
            "Cached solver can only accept RHS with the same shape "
            f"as the initialized value. {self.b_shape}. "
            "The initialized RHS shape can be changed by initializing "
            "using a different value for n_batches"
        )
        b = self._prepare_rhs(b)
        super().reset_operands(b=b)
        return super().solve()

    # Optional helpers for SuperLU-like API
    def __call__(self, b):  # x = solver(b)
        return self.solve(b)

    def refactorize_values(self, A_new):
        """Refactorize when numerical values change but sparsity pattern is the same."""
        if not isinstance(A_new, sp.csr_matrix):
            raise TypeError("A_new must be CSR")
        if A_new.shape != self.operands.a_shape:
            raise ValueError("Shape mismatch")
        if A_new.indices.dtype != cp.int32 or A_new.indptr.dtype != cp.int32:
            A_new = sp.csr_matrix((A_new.data, A_new.indices.astype(cp.int32),
                                   A_new.indptr.astype(cp.int32)), shape=A_new.shape)
        super().reset_operands(A=A_new)
        self.fac_info = self.factorize()

    def close(self):
        """Explicitly free resources when not using a context manager."""
        try:
            self.__exit__(None, None, None)  # DirectSolver is context-manageable
        except Exception:
            pass
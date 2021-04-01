import numpy as np

import xobjects as xo
from xobjects.context import available

def test_ffts():
    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        # Test on a square wave
        n_x = 1000
        x_host = np.zeros(n_x, dtype=np.complex128)
        x_host[:n_x//3] = 1.

        x_dev = ctx.nparray_to_context_array(x_host)
        myfft = ctx.plan_FFT(x_dev, axes=(0,))

        myfft.transform(x_dev)
        x_trans = ctx.nparray_from_context_array(x_dev).copy()

        myfft.itransform(x_dev)
        x_itrans = ctx.nparray_from_context_array(x_dev).copy()

        assert np.allclose(x_trans, np.fft.fft(x_host))
        assert np.allclose(x_itrans, x_host)


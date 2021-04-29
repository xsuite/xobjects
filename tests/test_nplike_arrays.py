import numpy as np

import xobjects as xo
from xobjects.context import available


def test_ffts():
    ctxs = [xo.ContextCpu, xo.ContextCupy]
    try:
        import gpyfft

        ctxs.append(
            xo.ContextPyopencl,
        )
    except ImportError:
        print("gpyfft not available")

    for CTX in ctxs:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        # Test on a square wave
        n_x = 1000
        x_host = np.zeros(n_x, dtype=np.complex128)
        x_host[: n_x // 3] = 1.0

        x_dev = ctx.nparray_to_context_array(x_host)
        myfft = ctx.plan_FFT(x_dev, axes=(0,))

        myfft.transform(x_dev)
        x_trans = ctx.nparray_from_context_array(x_dev).copy()

        myfft.itransform(x_dev)
        x_itrans = ctx.nparray_from_context_array(x_dev).copy()

        # Profit to test the extraction of the real part
        x_itrans = ctx.nparray_from_context_array(x_dev.real).copy()

        assert np.allclose(x_trans, np.fft.fft(x_host))
        assert np.allclose(x_itrans, x_host)


def test_slicing():
    for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
        if CTX not in available:
            continue

        print(f"Test {CTX}")
        ctx = CTX()

        for order in ("C", "F"):
            n_x = 100
            a_host = np.array(np.random.rand(n_x, n_x), order=order)

            a_dev = ctx.nparray_to_context_array(a_host)
            b_dev = a_dev[: n_x // 2, : n_x // 3]  # Non-countiguous array

            b_host = ctx.nparray_from_context_array(b_dev)
            assert np.allclose(b_host, a_host[: n_x // 2, : n_x // 3])

            # Test copy and setattr
            c_dev = a_dev.copy()[
                : n_x // 2, : n_x // 3
            ]  # Non-countiguous array
            c_dev[: n_x // 2 // 2, : n_x // 3 // 2] = (
                b_dev[: n_x // 2 // 2, : n_x // 3 // 2].copy() * 3
            )

            c_host = a_host.copy()[
                : n_x // 2, : n_x // 3
            ]  # Non-countiguous array
            c_host[: n_x // 2 // 2, : n_x // 3 // 2] = (
                b_host[: n_x // 2 // 2, : n_x // 3 // 2].copy() * 3
            )

            assert np.allclose(c_host, ctx.nparray_from_context_array(c_dev))

            # Check sum
            assert np.isclose(a_dev.sum(), a_host.sum())
            assert np.isclose(a_dev[:].sum(), a_host[:].sum())
            assert np.isclose(c_dev.sum(), c_host.sum())
            assert np.isclose(c_dev[:].sum(), c_host[:].sum())

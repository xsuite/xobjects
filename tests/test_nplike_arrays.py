# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo


def test_type():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

    a = ctx.zeros(10, dtype=np.float64)
    assert isinstance(a, ctx.nplike_array_type)


def test_ffts():

    for ctx in xo.context.get_test_contexts():
        if "Pyopencl" in str(ctx):
            try:
                import gpyfft
            except ImportError:
                print("gpyfft not available")
                continue

        print(f"Test {ctx}")

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
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

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


def test_nplike_from_xoarray():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        Array = xo.Float64[:]
        a_xo = Array(10, _context=ctx)
        a_nl = a_xo.to_nplike()
        a_nl[2] = 5
        assert a_xo[2] == 5

        Array = xo.Float64[10]
        a_xo = Array(_context=ctx)
        a_nl = a_xo.to_nplike()
        a_nl[2] = 5
        assert a_xo[2] == 5

        Array = xo.Float64[:, :]
        a_xo = Array(10, 20, _context=ctx)
        a_nl = a_xo.to_nplike()
        a_nl[2, 3] = 5
        assert a_xo[2, 3] == 5

        Array = xo.Float64[10, 20]
        a_xo = Array(_context=ctx)
        a_nl = a_xo.to_nplike()
        a_nl[2, 3] = 5
        assert a_xo[2, 3] == 5

        Array = xo.Int8[2:1, 3:0, 4:2]
        a_xo = Array(_context=ctx)
        assert a_xo._strides == (4, 8, 1)

        j = 0
        for i0, i1, i2 in np.ndindex(2, 3, 4):
            a_xo[i0, i1, i2] = j
            j += 1

        result = a_xo.to_nplike()
        expected = np.arange(2 * 3 * 4, dtype='int8').reshape((2, 3, 4))

        assert np.all(result == expected)
        assert result.strides == (4, 8, 1)


def test_nparray_from_xoarray():
    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx}")

        Array = xo.Float64[:]
        a_xo = Array(10, _context=ctx)
        a_xo[2] = 5
        a_nl = a_xo.to_nparray()
        assert a_nl[2] == 5

        Array = xo.Float64[10]
        a_xo = Array(_context=ctx)
        a_xo[2] = 5
        a_nl = a_xo.to_nparray()
        assert a_nl[2] == 5

        Array = xo.Float64[:, :]
        a_xo = Array(10, 20, _context=ctx)
        a_xo[2, 3] = 5
        a_nl = a_xo.to_nparray()
        assert a_nl[2, 3] == 5

        Array = xo.Float64[10, 20]
        a_xo = Array(_context=ctx)
        a_xo[2, 3] = 5
        a_nl = a_xo.to_nparray()
        assert a_nl[2, 3] == 5

        Array = xo.Int8[2:1, 3:0, 4:2]
        a_xo = Array(_context=ctx)
        assert a_xo._strides == (4, 8, 1)

        j = 0
        for i0, i1, i2 in np.ndindex(2, 3, 4):
            a_xo[i0, i1, i2] = j
            j += 1

        result = a_xo.to_nparray()
        expected = np.arange(2 * 3 * 4, dtype='int8').reshape((2, 3, 4))

        assert np.all(result == expected)
        assert result.strides == (4, 8, 1)

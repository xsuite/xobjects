# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
from numpy.testing import assert_allclose as np_assert_allclose
import numpy as np


class Print:
    suppress = False

    def __call__(self, *args, **kwargs):
        if not self.suppress:
            print(*args, **kwargs)


_print = Print()


def assert_allclose(a, b, rtol=0, atol=0, max_outliers=0):
    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "get"):
        b = b.get()
    try:
        a = np.squeeze(a)
    except:
        pass
    try:
        b = np.squeeze(b)
    except:
        pass

    try:
        np_assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        if max_outliers == 0:
            raise e
        if not allclose_with_outliers(a, b, rtol, atol, max_outliers):
            raise AssertionError(
                "Arrays are not close enough, even with outliers allowed."
            ) from e


def allclose_with_outliers(a, b, rtol=1e-7, atol=0, max_outliers=0):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    diff = np.abs(a - b)
    allowed = atol + rtol * np.abs(b)
    mask = diff > allowed
    num_outliers = np.count_nonzero(mask)
    return num_outliers <= max_outliers

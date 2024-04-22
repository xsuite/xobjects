# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
from numpy.testing import assert_allclose as np_assert_allclose


class Print:
    suppress = False

    def __call__(self, *args, **kwargs):
        if not self.suppress:
            print(*args, **kwargs)


_print = Print()


def assert_allclose(a, b, rtol=1e-7, atol=1e-7):
    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "get"):
        b = b.get()
    np_assert_allclose(a, b, rtol=rtol, atol=atol)

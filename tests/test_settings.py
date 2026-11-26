# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import xobjects as xo
import pytest


def test_cuda_compute_capability():
    assert xo.ContextCupy().cuda_compute_capability is None

    with xo.settings.override(cuda_compute_capability=90):
        assert xo.ContextCupy().cuda_compute_capability == 90
        assert (
            xo.ContextCupy(cuda_compute_capability=80).cuda_compute_capability
            == 80
        )

    with pytest.raises(ValueError):
        xo.ContextCupy(cuda_compute_capability=0)

    with pytest.raises(ValueError):
        xo.ContextCupy(cuda_compute_capability=-1)

    with pytest.raises(ValueError):
        xo.ContextCupy(cuda_compute_capability="90")

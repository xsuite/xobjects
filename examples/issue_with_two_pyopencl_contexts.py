# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo

ctx = xo.ContextPyopencl()
a = ctx.zeros((5, 5), dtype=np.float64)
b = a[:2:, :2]
c = b.copy()

print("First done")
ctx2 = xo.ContextPyopencl()
aa = ctx2.zeros((5, 5), dtype=np.float64)
bb = aa[:2:, :2]
cc = bb.copy()

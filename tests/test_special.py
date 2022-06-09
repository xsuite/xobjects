# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo

ctx = xo.ContextPyopencl()

buff = ctx.new_buffer(16)

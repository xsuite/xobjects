# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Field(xo.Float64, default=3)
    fb = xo.Field(xo.Float64)


class StructB(api.struct):
    fa = xo.Field(StructA, default=3)
    fb = xo.Field(xo.Float64[:], default=[1])


class StructC(api.struct):
    fa = xo.Field(StructA, default=3)
    fb = xo.Field(StructA[:], default=[])


class Array1(xo.Float64[:, 3]):
    pass


a = StructA()

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time
import xobjects as xo


class MyStruct(xo.Struct):
    a0 = xo.Float64[:]
    a1 = xo.Float64[:]
    a2 = xo.Float64[:]
    a3 = xo.Float64[:]
    a4 = xo.Float64[:]
    a5 = xo.Float64[:]
    a6 = xo.Float64[:]
    a7 = xo.Float64[:]
    a8 = xo.Float64[:]
    a9 = xo.Float64[:]
    a10 = xo.Float64[:]
    a11 = xo.Float64[:]
    a12 = xo.Float64[:]
    a13 = xo.Float64[:]
    a14 = xo.Float64[:]
    a15 = xo.Float64[:]
    a16 = xo.Float64[:]
    a17 = xo.Float64[:]
    a18 = xo.Float64[:]
    a19 = xo.Float64[:]


n_elements = 1000000

dic = {
    "a0": n_elements,
    "a1": n_elements,
    "a2": n_elements,
    "a3": n_elements,
    "a4": n_elements,
    "a5": n_elements,
    "a6": n_elements,
    "a7": n_elements,
    "a8": n_elements,
    "a9": n_elements,
    "a10": n_elements,
    "a11": n_elements,
    "a12": n_elements,
    "a13": n_elements,
    "a14": n_elements,
    "a15": n_elements,
    "a16": n_elements,
    "a17": n_elements,
    "a18": n_elements,
    "a19": n_elements,
}


t1 = time.time()
MyStruct._inspect_args(**dic)
t2 = time.time()
print(f"Time 'MyStruct._inspect_args(**dic)': {t2-t1:.9f} s")

t1 = time.time()
struct = MyStruct(**dic)
t2 = time.time()
print(f"Time to allocate the object: {t2-t1:.9f} s")

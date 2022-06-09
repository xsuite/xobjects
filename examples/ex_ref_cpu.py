# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Float64
    fb = xo.Int64[:]


class StructB(xo.Struct):
    fa = xo.Float64
    fb = xo.Ref[xo.Int64[:]]


ctx = xo.ContextCpu()


sa = StructA(fa=3, fb=4, _context=ctx)
sb = StructB(fa=3, _context=ctx)
sb.fb = [1, 2, 3, 4]

sa.fb[3] = 1
sb.fb[3] = 1

ks = {
    "ka": xo.Kernel([xo.Arg(StructA, name="obj")]),
    "kb": xo.Kernel([xo.Arg(StructB, name="obj")]),
}

source = """
void ka(StructA obj){
    printf("hello\\n");
    printf("fa=%g\\n", StructA_get_fa(obj));
    printf("len(fb)=%ld\\n", StructA_len_fb(obj));
    printf("fb[3]=%ld\\n", StructA_get_fb(obj,3));
};
void kb(StructB obj){
    printf("hello\\n");
    printf("fa=%g\\n", StructB_get_fa(obj));
    printf("len(fb)=%ld\\n", StructB_len_fb(obj));
    printf("fb[3]=%ld\\n", StructB_get_fb(obj,3));
};
"""

ctx.add_kernels([source], ks, extra_headers=["#include <stdio.h>"])

ctx.kernels.ka(obj=sa)
ctx.kernels.kb(obj=sb)

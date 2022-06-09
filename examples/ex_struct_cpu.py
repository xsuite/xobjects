# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Float64
    fb = xo.Int64[:]


source = """
void k1(StructA obj){
    printf("hello\\n");
    printf("fa=%g\\n", StructA_get_fa(obj));
    printf("len(fb)=%ld\\n", StructA_len_fb(obj));
    printf("fb[3]=%ld\\n", StructA_get_fb(obj,3));
};
double k2(StructA obj){
    return StructA_get_fa(obj)*3;
};
"""

ks = {
    "k1": xo.Kernel([xo.Arg(StructA, name="obj")]),
    "k2": xo.Kernel([xo.Arg(StructA, name="obj")], ret=xo.Arg(xo.Float64)),
}

ctx = xo.ContextCpu()
ctx.add_kernels([source], ks, extra_headers=["#include <stdio.h>"])

s1 = StructA(fa=0, fb=4, _context=ctx)
s1.fa = 2.3
s1.fb[3] = 1

ctx.kernels.k1(obj=s1)
ctx.kernels.k2(obj=s1)

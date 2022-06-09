# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Float64


class StructB(xo.Struct):
    fa = xo.Int64


class Ref(xo.UnionRef):
    _reftypes = (StructA, StructB)


ArrNRef = Ref[:]

ctx = xo.ContextCpu()

ar = ArrNRef(5, _context=ctx)

ar[0] = StructA(fa=3.1)
ar[1] = StructB(fa=5)

ks = {
    "kr": xo.Kernel([xo.Arg(ArrNRef, name="obj")]),
}

source = """
void kr(ArrNRef obj){
    printf("hello\\n");
    printf("typeid(ar[0])=%ld\\n", ArrNRef_typeid(obj,0));
    printf("typeid(ar[1])=%ld\\n", ArrNRef_typeid(obj,1));
    StructA s1 = (StructA) ArrNRef_member(obj,0);
    StructB s2 = (StructB) ArrNRef_member(obj,1);
    printf("ar[0].fa=%g\\n", StructA_get_fa(s1));
    printf("ar[1].fa=%ld\\n", StructB_get_fa(s2));

};
"""

ctx.add_kernels([source], ks, extra_headers=["#include <stdio.h>"])

ctx.kernels.kr(obj=ar)

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Float64
    fb = xo.Ref[xo.Int64[:]]


class StructB(xo.Struct):
    fa = xo.Float64
    fb = xo.Int64[:]


class ArrayB(xo.Float64[5]):
    pass


class RefA(xo.UnionRef):
    _reftypes = (StructA, ArrayB)


ArrNRefA = RefA[:]


ctx = xo.ContextCpu()


val1 = StructA(fa=3, fb=4, _context=ctx)
val2 = ArrayB(_context=ctx)
aref = RefA(val1, _buffer=val1._buffer)

aref = RefA(_context=ctx)

arr = ArrNRefA(10)

arr[0] = None
arr[1] = val1
arr[2] = val2


ks = {
    "k1": xo.Kernel([xo.Arg(StructA, name="obj")]),
    "k2": xo.Kernel([xo.Arg(StructA, name="obj")], ret=xo.Arg(xo.Float64)),
}

source = """
void k1(StructA obj){
    printf("hello\\n");
};
double k2(StructA obj){
    return StructA_get_fa(obj)*3;
};
"""

ctx.add_kernels(
    [source],
    ks,
    extra_headers=["#include <stdio.h>"],
    save_source_as="ex_unionref.c",
)

ctx.kernels.k1(obj=val1)
ctx.kernels.k2(obj=val1)

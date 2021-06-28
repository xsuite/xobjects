import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Float64
    fb = xo.Ref[xo.Int64]


class ArrayB(xo.Float64[5]):
    pass


class RefA(xo.UnionRef):
    _reftypes = (StructA, ArrayB)


ArrNRefA = RefA[:]

from xobjects.typeutils import sort_classes

sort_classes([ArrNRefA])

ctx = xo.ContextCpu()

aref = RefA(_context=ctx)

val1 = StructA(fa=3, _context=ctx)
val2 = ArrayB(_context=ctx)
aref = RefA(val1, _buffer=val1._buffer)

arr = ArrNRefA(10)

arr[0] = None
arr[1] = val1
arr[2] = val2


paths = ArrNRefA._gen_data_paths()

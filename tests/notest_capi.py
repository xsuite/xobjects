# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

# pylint:disable=E1101

import numpy as np
import xobjects as xo

from xobjects import capi
from xobjects.typeutils import default_conf


def gen_classes():
    class Field(xo.Struct):
        normal = xo.Float64
        skew = xo.Float64

    class Multipole(xo.Struct):
        order = xo.Int8
        angle = xo.Float64
        vlength = xo.Float64
        field = Field[:]

    return Field, Multipole


def test_gen_data_paths():
    Field, Multipole = gen_classes()
    Field_N = Multipole.field.ftype

    meth = Field._gen_data_paths()
    assert meth[0] == [Field.normal]
    assert meth[1] == [Field.skew]

    meth = Field_N._gen_data_paths()
    assert meth[0] == [Field_N]
    assert meth[1] == [Field_N, Field.normal]
    assert meth[2] == [Field_N, Field.skew]

    meth = Multipole._gen_data_paths()
    assert meth[0] == [Multipole.order]
    assert meth[1] == [Multipole.angle]
    assert meth[2] == [Multipole.vlength]
    assert meth[3] == [Multipole.field]
    assert meth[4] == [Multipole.field, Field_N]
    assert meth[5] == [Multipole.field, Field_N, Field.normal]
    assert meth[6] == [Multipole.field, Field_N, Field.skew]


def test_gen_get():
    Field, Multipole = gen_classes()
    Field_N = Multipole.field.ftype

    path = [Multipole.order]

    source, _ = capi.gen_method_get(Multipole, path, default_conf)
    assert (
        source
        == """\
/*gpufun*/ int8_t Multipole_get_order(const Multipole/*restrict*/ obj){
  int64_t offset=0;
  offset+=8;
  return *((/*gpuglmem*/int8_t*) obj+offset);
}"""
    )

    path = [Multipole.field, Field_N, Field.skew]
    source, _ = capi.gen_method_get(Multipole, path, default_conf)
    assert (
        source
        == """\
/*gpufun*/ double Multipole_get_field_skew(const Multipole/*restrict*/ obj, int64_t i0){
  int64_t offset=0;
  offset+=32;
  offset+=16+i0*16;
  offset+=8;
  return *(/*gpuglmem*/double*)((/*gpuglmem*/char*) obj+offset);
}"""
    )


def test_gen_set():
    _, Multipole = gen_classes()
    path = [Multipole.order]

    source, _ = capi.gen_method_set(Multipole, path, default_conf)
    assert (
        source
        == """\
/*gpufun*/ void Multipole_set_order(Multipole/*restrict*/ obj, int8_t value){
  int64_t offset=0;
  offset+=8;
  *((/*gpuglmem*/int8_t*) obj+offset)=value;
}"""
    )


def test_gen_c_api():
    _, Multipole = gen_classes()

    kernels = Multipole._gen_kernels()

    ctx = xo.ContextCpu()

    ctx.add_kernels(
        kernels=kernels,
    )

    m = Multipole(field=10)
    m.order = 3
    m.field[2].normal = 1
    assert ctx.kernels.Multipole_get_order(obj=m) == 3
    assert ctx.kernels.Multipole_get_field_normal(obj=m, i0=2) == 1.0


def test_ref():
    ctx = xo.ContextCpu()

    class StructA(xo.Struct):
        fa = xo.Float64
        sb = xo.Ref[xo.Int64[:]]

    paths = StructA._gen_data_paths()

    assert len(paths) == 4

    kernels = StructA._gen_kernels()

    ctx.add_kernels(kernels=kernels)

    sa = StructA(fa=2.3)
    sa.sb = [1, 2, 3]

    assert sa.fa == ctx.kernels.StructA_get_fa(obj=sa)
    assert sa.sb[2] == ctx.kernels.StructA_get_sb(obj=sa, i0=2)


def test_ref_union():
    class StructA(xo.Struct):
        fa = xo.Float64

    ArrayB = xo.Float64[6, 6]

    class RefA(xo.UnionRef):
        _reftypes = (StructA, ArrayB)

    ArrNRefA = RefA[:]

    paths = ArrNRefA._gen_data_paths()

    assert len(paths) == 2

    kernels = ArrNRefA._gen_kernels()

    arr = ArrNRefA(3)
    arr[0] = ("StructA", {"fa": 3.0})
    arr[1] = ArrayB()
    arr[1][2, 3] = 4.0

    ctx = xo.ContextCpu()
    ctx.add_kernels(kernels=kernels)

    assert ctx.kernels.ArrNRefA_typeid(obj=arr, i0=0) == 0
    assert ctx.kernels.ArrNRefA_typeid(obj=arr, i0=1) == 1
    assert ctx.kernels.ArrNRefA_typeid(obj=arr, i0=2) == -1


def test_capi_call():
    class ParticlesData(xo.Struct):
        s = xo.Int64[:]
        x = xo.Int64[:]
        y = xo.Int64[:]

    kernels = ParticlesData._gen_kernels()

    context = xo.ContextCpu()
    context.add_kernels(kernels=kernels)

    particles = ParticlesData(
        s=np.arange(10, 21, 10),
        x=np.arange(10, 21, 10),
        y=np.arange(10, 21, 10),
    )

    assert (
        context.kernels.ParticlesData_get_x(obj=particles, i0=1)
        == particles.x[1]
    )


def test_2_particles():
    context = xo.ContextCpu()

    class ParticlesData(xo.Struct):
        num_particles = xo.Int64
        s = xo.Float64[:]
        x = xo.Float64[:]

    particles = ParticlesData(
        num_particles=2, s=np.array([1, 2]), x=np.array([7, 8])
    )

    kernels = ParticlesData._gen_kernels()
    context.add_kernels(kernels=kernels)

    ptr = particles._buffer.to_nplike(0, "int64", (11,)).ctypes.data
    print(f"{ptr:x}")
    assert (
        context.kernels.ParticlesData_get_x(obj=particles, i0=0)
        == particles.x[0]
    )
    assert (
        context.kernels.ParticlesData_get_x(obj=particles, i0=1)
        == particles.x[1]
    )

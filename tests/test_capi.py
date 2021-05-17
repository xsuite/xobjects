# pylint:disable=E1101

import numpy as np
import xobjects as xo

from xobjects import capi


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
    assert meth[0] == [Field, Field.normal]
    assert meth[1] == [Field, Field.skew]

    meth = Field_N._gen_data_paths()
    meth[0] = [Field_N, Field_N]
    meth[1] = [Field_N, Field_N, Field.normal]
    meth[2] = [Field_N, Field_N, Field.skew]

    meth = Multipole._gen_data_paths()
    assert meth[0] == [Multipole, Multipole.order]
    assert meth[1] == [Multipole, Multipole.angle]
    assert meth[2] == [Multipole, Multipole.vlength]
    assert meth[3] == [Multipole, Multipole.field]
    assert meth[4] == [Multipole, Multipole.field, Field_N]
    assert meth[5] == [Multipole, Multipole.field, Field_N, Field.normal]
    assert meth[6] == [Multipole, Multipole.field, Field_N, Field.skew]


def test_gen_get():
    Field, Multipole = gen_classes()
    Field_N = Multipole.field.ftype

    path = [Multipole, Multipole.order]
    conf = {"gpumem": "/*gpuglmem*/ "}

    source, _ = capi.gen_method_get(path, conf)
    assert (
        source
        == """\
int8_t Multipole_get_order(const Multipole obj){
  int64_t offset=0;
  offset+=8;
  return *((/*gpuglmem*/ int8_t*) obj+offset);
}"""
    )

    path = [Multipole, Multipole.field, Field_N, Field.skew]
    source, _ = capi.gen_method_get(path, conf)
    assert (
        source
        == """\
double Multipole_get_field_skew(const Multipole obj, int64_t i0){
  int64_t offset=0;
  offset+=32;
  offset+=16+i0*16;
  offset+=8;
  return *(/*gpuglmem*/ double*)((/*gpuglmem*/ char*) obj+offset);
}"""
    )


def test_gen_set():
    _, Multipole = gen_classes()
    path = [Multipole, Multipole.order]
    conf = {"gpumem": "/*gpuglmem*/ "}

    source, _ = capi.gen_method_set(path, conf)
    assert (
        source
        == """\
void Multipole_set_order(Multipole obj, int8_t value){
  int64_t offset=0;
  offset+=8;
  *((/*gpuglmem*/ int8_t*) obj+offset)=value;
}"""
    )


def test_gen_c_api():
    _, Multipole = gen_classes()

    source, kernels, cdef = Multipole._gen_c_api()

    ctx = xo.ContextCpu()

    ctx.add_kernels(
        sources=[source],
        kernels=kernels,
        extra_cdef=cdef,
        save_source_as="test_get_c_api.c",
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

    assert len(paths) == 3

    source, kernels, cdef = StructA._gen_c_api()

    ctx.add_kernels(
        sources=[source],
        kernels=kernels,
        extra_cdef=cdef,
        specialize=True,
        save_source_as="test_ref.c",
    )


def test_ref_union():
    ctx = xo.ContextCpu()

    class StructA(xo.Struct):
        fa = xo.Float64

    ArrayB = xo.Float64[6, 6]

    class MyArray(xo.Ref[StructA, ArrayB][:]):
        pass

    paths = MyArray._gen_data_paths()

    assert len(paths) == 4

    source, kernels, cdef = MyArray._gen_c_api()

    ctx.add_kernels(
        sources=[source],
        kernels=kernels,
        extra_cdef=cdef,
        specialize=True,
        save_source_as="test_ref_union.c",
    )


def test_capi_call():
    class ParticlesData(xo.Struct):
        s = xo.Int64[:]
        x = xo.Int64[:]
        y = xo.Int64[:]

    source, kernels, cdefs = ParticlesData._gen_c_api()

    context = xo.ContextCpu()
    context.add_kernels(
        [source], kernels, extra_cdef=cdefs, save_source_as="test_capi_call.c"
    )

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

    source, kernels, cdefs = ParticlesData._gen_c_api()
    context.add_kernels([source], kernels, extra_cdef=cdefs)

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

    return particles

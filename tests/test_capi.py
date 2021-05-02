# pylint:disable=E1101


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
    assert meth[0] == [Field.normal]
    assert meth[1] == [Field.skew]

    meth = Field_N._gen_data_paths()
    meth[0] = [Field_N]
    meth[1] = [Field_N, Field.normal]
    meth[2] = [Field_N, Field.skew]

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

    parts = [Multipole.order]

    source, _ = capi.gen_method_get(Multipole, parts, {})
    assert (
        source
        == """\
int8_t Multipole_get_order(const Multipole obj){
  int64_t offset=0;
  offset+=8;
  return *((int8_t*) obj+offset);
}"""
    )

    parts = [Multipole.field, Field_N, Field.skew]
    source, _ = capi.gen_method_get(Multipole, parts, {})
    assert (
        source
        == """\
double Multipole_get_field_skew(const Multipole obj, int64_t i0){
  int64_t offset=0;
  offset+=32;
  offset+=16+i0*16;
  offset+=8;
  return *(double*)((char*) obj+offset);
}"""
    )


def test_gen_set():
    _, Multipole = gen_classes()
    parts = [Multipole.order]

    source, _ = capi.gen_method_set(Multipole, parts, {})
    assert (
        source
        == """\
void Multipole_set_order(Multipole obj, int8_t value){
  int64_t offset=0;
  offset+=8;
  *((int8_t*) obj+offset)=value;
}"""
    )


def test_gen_c_api():
    _, Multipole = gen_classes()

    ctx = xo.ContextCpu()

    source, kernels, cdef = Multipole._gen_c_api()

    ctx.add_kernels(
        sources=[source],
        kernels=kernels,
        extra_cdef=cdef,
        save_source_as="test.c",
    )

    m = Multipole(field=10)
    m.order = 3
    m.field[2].normal = 1
    assert ctx.kernels.Multipole_get_order(obj=m) == 3
    assert ctx.kernels.Multipole_get_field_normal(obj=m, i0=2) == 1.0


def notest_ref():
    class StructA(xo.Struct):
        fa = xo.Float64
        sb = xo.Ref[xo.Int64][:]

    ArrayB = xo.Float64[6, 6]

    class List(xo.Ref[StructA, ArrayB][:]):
        pass

    paths = List._gen_data_paths()

    assert len(paths) == 3

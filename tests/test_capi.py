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


def test_gen_method_spec():

    Field, Multipole = gen_classes()

    meth = Field._gen_method_specs()
    assert meth[0] == [Field.normal]

    Field_N = Multipole.field.ftype
    meth = Field_N._gen_method_specs()
    meth[0] = [Field_N]
    meth[1] = [Field_N, Field.normal]
    meth[2] = [Field_N, Field.skew]

    meth = Multipole._gen_method_specs()
    assert meth[0] == [Multipole.order]
    assert meth[1] == [Multipole.angle]
    assert meth[2] == [Multipole.vlength]
    assert meth[3] == [Multipole.field]
    assert meth[4] == [Multipole.field, Field_N]
    assert meth[5] == [Multipole.field, Field_N, Field.normal]
    assert meth[6] == [Multipole.field, Field_N, Field.skew]


def test_gen_get():
    _, Multipole = gen_classes()
    parts = [Multipole.order]

    code = capi.gen_get(Multipole, parts, {})
    assert (
        code
        == """\
int8_t Multipole_get_order(const Multipole obj){
  int64_t offset=0;
  offset+=8;
  return *((int8_t*) obj+offset);
}"""
    )

    ctx = xo.ContextCpu()

    source, kernels = Multipole._gen_c_api()

    ctx.add_kernels(sources=[source], kernels=kernels)


def test_gen_set():
    _, Multipole = gen_classes()
    parts = [Multipole.order]

    code = capi.gen_set(Multipole, parts, {})
    assert code == "void Multipole_set_order(Multipole obj, int8_t value);"


def test_gen_getp():
    _, Multipole = gen_classes()
    parts = [Multipole.order]

    code = capi.gen_getp(Multipole, parts, {})
    assert code == "int8_t* Multipole_getp_order(const Multipole obj);"


# def test_gen_method_get_declaration():

#     Field, Multipole = gen_classes()

#     methods = Field._gen_method_get_declaration()

#     assert methods[0] == "double Field_get_normal(Field* obj)"
#     assert methods[1] == "double Field_get_skew(Field* obj)"

#     methods = Multipole._gen_method_get_declaration()

#     assert methods[0] == "int8_t Multipole_get_order(Multipole* obj)"
#     assert methods[1] == "double Multipole_get_angle(Multipole* obj)"
#     assert methods[2] == "double Multipole_get_vlength(Multipole* obj)"
#     assert methods[3] == "Field_N* Multipole_get_field(Multipole* obj)"
#     assert (
#         methods[4] == "Field* Multipole_get_field(Multipole* obj, int64_t i0)"
#     )
#     assert (
#         methods[5]
#         == "double Multipole_get_field_normal(Multipole* obj, int64_t i0)"
#     )
#     assert (
#         methods[6]
#         == "double Multipole_get_field_skew(Multipole* obj, int64_t i0)"
#     )


# def test_get_method_offset():
#     Field, Multipole = gen_classes()

#     assert Field.normal._get_c_offset({}) == 0
#     assert Field.skew._get_c_offset({}) == 8

#     assert Multipole.order._get_c_offset({}) == 8

#     Field_N = Multipole.field.ftype

#     assert Field_N._get_c_offset({}) == ["  offset+=16+i0*16;"]


# def test_gen_method_get_body():

#     Field, Multipole = gen_classes()

#     methods = Field._gen_method_get_definition()

#     assert (
#         methods[0]
#         == """\
# double Field_get_normal(Field* obj){
#   int64_t offset=0;
#   return *(double*)((char*) obj+offset);
# }"""
#     )
#     assert (
#         methods[1]
#         == """\
# double Field_get_skew(Field* obj){
#   int64_t offset=0;
#   offset+=8;
#   return *(double*)((char*) obj+offset);
# }"""
#     )

#     methods = Multipole._gen_method_get_definition()

#     assert (
#         methods[0]
#         == """\
# int8_t Multipole_get_order(Multipole* obj){
#   int64_t offset=0;
#   offset+=8;
#   return *((int8_t*) obj+offset);
# }"""
#     )

#     assert (
#         methods[1]
#         == """\
# double Multipole_get_angle(Multipole* obj){
#   int64_t offset=0;
#   offset+=16;
#   return *(double*)((char*) obj+offset);
# }"""
#     )

#     assert (
#         methods[2]
#         == """\
# double Multipole_get_vlength(Multipole* obj){
#   int64_t offset=0;
#   offset+=24;
#   return *(double*)((char*) obj+offset);
# }"""
#     )

#     assert (
#         methods[3]
#         == """\
# Field_N* Multipole_get_field(Multipole* obj){
#   int64_t offset=0;
#   offset+=32;
#   return (Field_N*)((char*) obj+offset);
# }"""
#     )

#     assert (
#         methods[4]
#         == """\
# Field* Multipole_get_field(Multipole* obj, int64_t i0){
#   int64_t offset=0;
#   offset+=32;
#   offset+=16+i0*16;
#   return (Field*)((char*) obj+offset);
# }"""
#     )

#     assert (
#         methods[5]
#         == """\
# double Multipole_get_field_normal(Multipole* obj, int64_t i0){
#   int64_t offset=0;
#   offset+=32;
#   offset+=16+i0*16;
#   return *(double*)((char*) obj+offset);
# }"""
#     )

#     assert (
#         methods[6]
#         == """\
# double Multipole_get_field_skew(Multipole* obj, int64_t i0){
#   int64_t offset=0;
#   offset+=32;
#   offset+=16+i0*16;
#   offset+=8;
#   return *(double*)((char*) obj+offset);
# }"""
#     )


# def test_struct_getter():
#     class AStruct(xo.Struct):
#         fa = xo.Int64
#         fb = xo.Float64

#     ctx = xo.ContextCpu()

#     source = AStruct._get_c_api()

#     ctx.add_kernels

#     assert source == source

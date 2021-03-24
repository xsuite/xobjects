import xobjects as xo


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
    assert meth[3] == [Multipole.field]
    assert meth[4] == [Multipole.field, Field_N]
    assert meth[5] == [Multipole.field, Field_N, Field.normal]
    assert meth[6] == [Multipole.field, Field_N, Field.skew]


def test_gen_methods():

    Field, Multipole = gen_classes()

    methods = Field._gen_methods()

    assert methods[0] == "double Field_get_normal(Field* obj)"
    assert methods[1] == "double Field_get_skew(Field* obj)"

    methods = Multipole._gen_method_def()

    assert methods[0] == "int8_t Multipole_get_order(Multipole* obj)"
    assert methods[1] == "double Multipole_get_angle(Multipole* obj)"
    assert methods[2] == "double Multipole_get_vlength(Multipole* obj)"
    assert methods[3] == "Field_N* Multipole_get_field(Multipole* obj)"
    assert (
        methods[4] == "Field* Multipole_get_field(Multipole* obj, int64_t i0)"
    )
    assert (
        methods[5]
        == "double Multipole_get_field_normal(Multipole* obj, int64_t i0)"
    )
    assert (
        methods[6]
        == "double Multipole_get_field_skew(Multipole* obj, int64_t i0)"
    )

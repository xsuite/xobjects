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


def test_struct_simple():

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

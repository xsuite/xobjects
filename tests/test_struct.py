import xobjects as xo

def test_static_struct():
    class StructA(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(xo.Int8,default=-4)
        c=xo.Field(xo.Int64,default=-1)

    assert StructA.a.index==0
    assert StructA.b.index==1
    assert StructA.c.index==2

    s=StructA()
    assert s.a==3.5
    assert s.b==-4
    assert s.c==-1

    s.a=5.2
    assert s.a==5.2
    s.c=7
    assert s.c==7
    s.b=-4


def test_nested_struct():
    class StructB(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(xo.Int64,default=-4)
        c=xo.Field(xo.Int8,default=-1)

    class StructC(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(StructB)
        c=xo.Field(xo.Int8,default=-1)

    b=StructC()
    assert b.a==3.5
    assert b.b.a==3.5


def test_dynamic_struct():
    class StructD(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(xo.String,default=10)
        c=xo.Field(xo.Int8,default=-1)


    d=StructD(b="this is a test")
    assert d.a==3.5
    assert d.b=="this is a test"
    assert d.c==-1


def test_dynamic_nested_struct():
    class StructE(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(xo.String,default=10)
        c=xo.Field(xo.Int8,default=-1)

    class StructF(xo.Struct):
        a=xo.Field(xo.Float64,default=3.5)
        b=xo.Field(StructE)
        c=xo.Field(xo.Int8,default=-1)

    f=StructF(b={"b": "this is a test"})
    assert f.a==3.5
    assert f.b.b=="this is a test"
    assert f.c==-1




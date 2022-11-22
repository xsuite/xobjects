def test_to_json():
    import xobjects as xo

    class A(xo.Struct):
        a = xo.Float64[:]

    class B(xo.Struct):
        c = xo.Float64[:]
        d = xo.Int64

    class Uref(xo.UnionRef):
        _reftypes = A, B

    AUref = Uref[:]

    a = AUref(10)
    a[1] = A(a=[3])
    a[5] = B(c=2, d=1)

    b = AUref(a._to_json())

    assert b[1].a[0] == 3
    assert b[5].d == 1

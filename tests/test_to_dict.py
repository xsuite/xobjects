import xobjects as xo


def test_to_dict():
    class A(xo.Struct):
        a = xo.Float64[:]
        b = xo.Int64

    class Uref(xo.UnionRef):
        _reftypes = (A,)

    x = A(a=[2, 3], b=1)
    u = Uref(x)
    v = Uref(*u._to_dict())

    assert v.get().a[0] == 2
    assert v.get().a[1] == 3
    assert v.get().b == 1


def test_to_dict_array():
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

    b = AUref(a._to_dict())

    assert b[1].a[0] == 3
    assert b[5].d == 1


def test_to_dict_array_multidimensional_static_shape():
    array_type = xo.Float64[2, 3]
    array = array_type([[1, 2, 3], [4, 5, 6]])

    assert array._to_dict() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    rebuilt = array_type(array._to_dict())
    assert rebuilt._to_dict() == array._to_dict()


def test_to_dict_array_multidimensional_dynamic_shape():
    array_type = xo.Float64[:, :]
    array = array_type(2, 3)
    array[0, 0] = 1
    array[0, 1] = 2
    array[0, 2] = 3
    array[1, 0] = 4
    array[1, 1] = 5
    array[1, 2] = 6

    assert array._to_dict() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    rebuilt = array_type(array._to_dict())
    assert rebuilt._to_dict() == array._to_dict()


def test_to_dict_array_of_structs():
    class Item(xo.Struct):
        value = xo.Int64
        coords = xo.Float64[2]

    array_type = Item[:]
    items = array_type(
        [
            {"value": 3, "coords": [1, 2]},
            {"value": 7, "coords": [4, 5]},
        ]
    )

    assert items._to_dict() == [
        {"value": 3, "coords": [1.0, 2.0]},
        {"value": 7, "coords": [4.0, 5.0]},
    ]
    rebuilt = array_type(items._to_dict())
    assert rebuilt._to_dict() == items._to_dict()


def test_to_dict_multidimensional_array_of_structs():
    class Item(xo.Struct):
        value = xo.Int64
        coords = xo.Float64[2]

    array_type = Item[:, :]
    items = array_type(
        [
            [
                {"value": 1, "coords": [1, 2]},
                {"value": 2, "coords": [3, 4]},
            ],
            [
                {"value": 3, "coords": [5, 6]},
                {"value": 4, "coords": [7, 8]},
            ],
        ]
    )

    expected = [
        [
            {"value": 1, "coords": [1.0, 2.0]},
            {"value": 2, "coords": [3.0, 4.0]},
        ],
        [
            {"value": 3, "coords": [5.0, 6.0]},
            {"value": 4, "coords": [7.0, 8.0]},
        ],
    ]

    assert items._to_dict() == expected
    rebuilt = array_type(items._to_dict())
    assert rebuilt._to_dict() == expected


def test_to_dict_multidimensional_array_of_structs_with_refs():
    class Item(xo.Struct):
        values = xo.Ref[xo.Float64[:]]
        weight = xo.Int64

    array_type = Item[:, :]
    items = array_type(
        [
            [
                {"values": [1, 2], "weight": 3},
                {"values": [4, 5, 6], "weight": 7},
            ],
            [
                {"values": [8], "weight": 9},
                {"values": [10, 11], "weight": 12},
            ],
        ]
    )

    expected = [
        [
            {"values": [1.0, 2.0], "weight": 3},
            {"values": [4.0, 5.0, 6.0], "weight": 7},
        ],
        [
            {"values": [8.0], "weight": 9},
            {"values": [10.0, 11.0], "weight": 12},
        ],
    ]

    assert items._to_dict() == expected
    rebuilt = array_type(items._to_dict())
    assert rebuilt._to_dict() == expected

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class StructA(xo.Struct):
    fa = xo.Field(xo.float64, default=0)


# shortcut to be considered
@xo.struct
class StructA:
    fa: xo.Float64 = 0


@xo.struct
class StructB:
    fa: xo.Float64[6, 6]
    fb: StructA[None]


class StructC:
    def _parse_args_(self, *args, **kwargs):
        return args, kwargs


a = StructA()  # default struct
a = Struct({"fa": 0})  # create and  assign with dict
a = Struct(fa=3)  # create and assign with arg
b = Struct(a)  # create and assign with same instance
d = a.to_dict()  # create dict

b = StructB(fb=[{"fa": 0}])

ctx = CLContext(device="0.0")
c = StructA(
    b, _context=ctx
)  # create buffer on ctx and initialize with compatible object
c = StructA(
    b, _buffer=buff
)  # allocate on buffer and initialize compatible object
c = StructA(
    b, _buffer=buff, _offset=10
)  # use buffer from offset and initialize with compatible object


class Field(xo.Struct):
    normal = xo.Field(xo.Float64[10])
    skew = xo.Field(xo.Float64[10])


Fields = Field[:]


class Multipole(xo.Struct):
    order = xo.Field(xo.Int64)
    field = Fields


class Element(xo.Union):
    _members = [Multipole, Drift]


class Line(xo.Array):
    _member = Element
    _shape = None


l = Line()

l.append(Multipole(field=[{"normal": 0, "skew": 0}] * 10))
l.append(Multipole(field=10))
l.append(Multipole(field=Fields(10)))
l.append(Multipole(field=Fields([0, 0] * 10)))

k = Multipole_get_field_normal_test(mult, 0, 3)

c = StructA(buff, b)  # allocate on buffer and initialize compatible object
c = StructA(
    View(buff, 10), b
)  # use buffer from offset and initialize with compatible object

c = Struct.new(
    ctx, b
)  # create buffer on ctx and initialize with compatible object
c = Struct.new(buff, b)  # allocate on buffer and initialize compatible object
c = Struct.new(
    View(buff, 10), b
)  # use buffer from offset and initialize with compatible object


c = Struct(_context=ctx)  # create buffer on ctx and initialized default objecy
c = Struct(_buffer=buff)  # allocate on buffer and copy compatible object
c = Struct(
    _buffer=buff, _offset=10
)  # use buffer from offset and copy compatible object

c = b.copy(ctx)  # create on ctx and assign with compatible object

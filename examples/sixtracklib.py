# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""
struct Particle x f64 px f64 y f64 py f64
array Particles Particle64 :

struct Drift length f64

struct Field normal f64 skew f64
array  Fields Field :
struct Multipole order i64 angle f64 length f64 field Fields

array ClosedOrbit f64 6
array SigmaMatrix f64 6 6
array Position f64 3
array Kick f64 6
struct BeamBeam6D charge f64 location Position sigma SigmaMatrix weak_orbit ClosedOrbit  weak_kick Kick


union Element Drift Multipole BeamBeam6D
array Elements Element :
"""


import xobject as xo


class Particle(xo.Struct):
    x = xo.Float64
    px = xo.Float64
    y = xo.Float64
    py = xo.Float64


class Particles(Particle[:]):
    pass


class Particles(xo.SOA):
    _itemtype = Particle
    _shape = [None]


# or
class Particles(Particle.soa[:]):
    pass


class Drift(xo.Struct):
    length = xo.Float64


class Field(xo.Struct):
    normal = xo.Float64
    skew = xo.Float64


class Multipole(xo.Struct):
    order = xo.Int8
    angle = xo.Float64
    vlength = xo.Float64
    field = Field[:]


class BeamBeam6D(xo.Struct):
    charge = xo.Float64
    location = xo.Float64[3]
    sigma = xo.Float64[6, 6]
    weak_orbit = xo.Float64[6]
    correction = xo.Float64[6]


class Element(xo.Union):
    _members = (Drift, Multipole, BeamBeam6D)


class Elements(Element[:]):
    pass

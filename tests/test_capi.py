import xobjects as xo


def test_struct():
    class Particle(xo.Struct):
        x = xo.Float64
        px = xo.Float64
        y = xo.Float64
        py = xo.Float64

    assert Particle._generate_methods() == [
        [Particle.x],
        [Particle.px],
        [Particle.y],
        [Particle.py],
    ]

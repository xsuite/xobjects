# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo


class Triangle(xo.Struct):
    b = xo.Float64
    h = xo.Float64

    _extra_c_sources = ["""
    /*gpufun*/
    double Triangle_compute_area(Triangle tr, double scale){
        double b = Triangle_get_b(tr);
        double h = Triangle_get_h(tr);
        return 0.5*b*h*scale;
    }
    """]


class Square(xo.Struct):
    a = xo.Float64

    _extra_c_sources = ["""
    /*gpufun*/
    double Square_compute_area(Square sq, double scale){
        double a = Square_get_a(sq);
        return a*a*scale;
    }
    """]


class Base(xo.UnionRef):
    _reftypes = (Triangle, Square)
    _methods = [
        xo.Method(
            c_name="compute_area",
            args=[xo.Arg(xo.Float64, name="scale")],
            ret=xo.Arg(xo.Float64),
        )
    ]


class Prism(xo.Struct):
    base = Base
    height = xo.Float64
    volume = xo.Float64

    _extra_c_sources = ["""
    /*gpukern*/
    void Prism_compute_volume(Prism pr){
        Base base = Prism_getp_base(pr);
        double height = Prism_get_height(pr);
        double base_area = Base_compute_area(base, 3.);
        Prism_set_volume(pr, base_area*height);
    }
    """]


context = xo.ContextCpu()
context.add_kernels(
    kernels={
        "Prism_compute_volume": xo.Kernel(args=[xo.Arg(Prism, name="prism")])
    }
)


triangle = Triangle(b=2, h=3)
prism_triangle = Prism(base=triangle, height=5)
square = Square(a=2)
prism_square = Prism(base=square, height=10)

context.kernels.Prism_compute_volume(prism=prism_triangle)
context.kernels.Prism_compute_volume(prism=prism_square)

assert prism_triangle.volume == 45
assert prism_square.volume == 120


# OpenCL
context = xo.ContextPyopencl()
context.add_kernels(
    kernels={
        "Prism_compute_volume": xo.Kernel(args=[xo.Arg(Prism, name="prism")])
    }
)


triangle = Triangle(b=2, h=3, _context=context)
prism_triangle = Prism(base=triangle, height=5, _context=context)
square = Square(a=2, _context=context)
prism_square = Prism(base=square, height=10, _context=context)

context.kernels.Prism_compute_volume(prism=prism_triangle)
context.kernels.Prism_compute_volume(prism=prism_square)

assert prism_triangle.volume == 45
assert prism_square.volume == 120

# Cuda
context = xo.ContextCupy()
context.add_kernels(
    kernels={
        "Prism_compute_volume": xo.Kernel(args=[xo.Arg(Prism, name="prism")])
    }
)


triangle = Triangle(b=2, h=3, _context=context)
prism_triangle = Prism(base=triangle, height=5, _context=context)
square = Square(a=2, _context=context)
prism_square = Prism(base=square, height=10, _context=context)

context.kernels.Prism_compute_volume(prism=prism_triangle)
context.kernels.Prism_compute_volume(prism=prism_square)

assert prism_triangle.volume == 45
assert prism_square.volume == 120

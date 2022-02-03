import xobjects as xo

class Triangle(xo.Struct):
    b = xo.Float64
    h = xo.Float64

Triangle.extra_sources = [
    '''
    double Triangle_compute_area(Triangle tr){
        double b = Triangle_get_b(tr);
        double h = Triangle_get_h(tr);
        return 0.5*b*h;
    }
    '''
]

class Square(xo.Struct):
    a = xo.Float64

Square.extra_sources = [
    '''
    double Square_compute_area(Square sq){
        double a = Square_get_a(sq);
        return a*a;
    }
    '''
]

class Base(xo.UnionRef):
    _reftypes = (Triangle, Square)

Base.extra_sources =[
    '''
    double Base_compute_area(Base base){
        void* member = Base_member(base);
        switch (Base_typeid(base)){
            #ifndef BASE_SKIP_TRIANGLE
            case Base_Triangle_t:
                return Triangle_compute_area((Triangle) member);
                break;
            #endif
            #ifndef BASE_SKIP_SQUARE
            case Base_Square_t:
                return Square_compute_area((Square) member);
                break;
            #endif
        }
        return 0;
    }
    '''
]

class Prism(xo.Struct):
    base = Base
    height = xo.Float64
    volume = xo.Float64

Prism.extra_sources = [
    '''
    void Prism_compute_volume(Prism pr){
        Base base = Prism_getp_base(pr);
        double height = Prism_get_height(pr);

        double base_area = Base_compute_area(base);

        Prism_set_volume(pr, base_area*height);
    }
    '''
]

context = xo.ContextCpu()
context.add_kernels(sources=(Triangle.extra_sources + Square.extra_sources +
                             Base.extra_sources + Prism.extra_sources),
    kernels = {'Prism_compute_volume': xo.Kernel(
        args = [xo.Arg(Prism, name='prism')])})

triangle = Triangle(b=2, h=3)
prism_triangle = Prism(base=triangle, height=5)
square = Square(a=2)
prism_square = Prism(base=square, height=10)

context.kernels.Prism_compute_volume(prism=prism_triangle)
context.kernels.Prism_compute_volume(prism=prism_square)
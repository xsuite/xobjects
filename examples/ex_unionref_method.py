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


class Prism(xo.Struct):
    base = Triangle
    height = xo.Float64
    volume = xo.Float64

Prism.extra_sources = [
    '''
    void Prism_compute_volume(Prism pr){
        Triangle base = Prism_getp_base(pr);
        double height = Prism_get_height(pr);

        double base_area = Triangle_compute_area(base);

        Prism_set_volume(pr, base_area*height);
    }
    '''
]

context = xo.ContextCpu()
import pdb; pdb.set_trace()
context.add_kernels(sources=(Triangle.extra_sources + Prism.extra_sources),
    kernels = {'Prism_compute_volume': xo.Kernel(
        args = [xo.Arg(Prism, 'prism')])})

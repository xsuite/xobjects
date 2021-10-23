import xobjects as xo

context = xo.ContextCpu()

class MyStruct(xo.Struct):
    a = xo.Float64[:]

class MyStruct2(xo.Struct):
    a = xo.Float64[:]
    sr = xo.Ref(MyStruct)

ms = MyStruct(a=[1,2,3], _context=context)

ms2 = MyStruct2(_buffer=ms._buffer, sr=ms, a=[0,0,0])

src = '''
void cp_sra_to_a(MyStruct2 ms, int64_t n){

    for(int64_t ii=0; ii<n; ii++){
        double const val = MyStruct2_get_sr_a(ms, ii);
        MyStruct2_set_a(ms, ii, val);
    }

}
'''

context.add_kernels(sources=[src], kernels={
    'cp_sra_to_a': xo.Kernel(args=[
        xo.Arg(MyStruct2, name='ms'),
        xo.Arg(xo.Int64, name='n')])})

context.kernels.cp_sra_to_a(ms=ms2, n=len(ms.a))

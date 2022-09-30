import xobjects as xo
import numpy as np
import time


def timeit(code, nn=200):
    import math

    cc = compile(code, "", "exec")
    t1 = time.time()
    for i in range(nn):
        exec(cc)
    t2 = time.time()
    dt = (t2 - t1) / nn
    oo = int(math.log10(dt) / 3)-1
    unit = "num kM"[oo + 3]
    scale = 1000**oo
    print(f"{nn:4} times {dt/scale:8.3f} {unit}sec  {code}")


Arr = xo.Float64[:]


class StA(xo.Struct):
    a = Arr


class StS(xo.Struct):
    s = xo.Float64


buf = xo.context_default.new_buffer(2**20)
al = [1, 2, 3]
ar = np.array(al)


timeit("Arr(al,_buffer=buf)", 1000)
timeit("Arr(ar,_buffer=buf)", 1000)

timeit("StS(s=1,_buffer=buf)", 1000)
timeit("StA(a=al,_buffer=buf)", 1000)
timeit("StA(a=ar,_buffer=buf)", 1000)

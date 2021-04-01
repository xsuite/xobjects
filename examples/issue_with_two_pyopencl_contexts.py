import numpy as np
import xobjects as xo

ctx = xo.ContextPyopencl()
a = ctx.zeros((5,5), dtype=np.float64)
b = a[:2:,:2]
c = b.copy()

print('Fists done')
ctx = xo.ContextPyopencl()
a = ctx.zeros((5,5), dtype=np.float64)
b = a[:2:,:2]
c = b.copy()

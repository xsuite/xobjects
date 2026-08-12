# Xobjects

Xobjects is a low-level Python package for defining structured data that can be
stored in CPU or GPU memory and accessed from generated C APIs. It is used by
the Xsuite ecosystem as the memory, serialization, and kernel-compilation layer,
but it can also be used directly when a project needs the same data layout from
Python and from compiled CPU, CUDA, or OpenCL code.

The main building blocks are:

- contexts, which select the execution and memory backend;
- buffers, which own memory allocated by a context;
- xobject data types, which describe the binary layout of scalars, arrays,
  structures, strings, and references;
- kernels, which bind C-like source code to typed Python callables.

## Installation

```bash
python -m pip install xobjects
```

For local development:

```bash
git clone https://github.com/xsuite/xobjects.git
cd xobjects
python -m pip install -e ".[tests]"
```

GPU contexts require their corresponding runtime packages and drivers:

- `ContextCpu` uses NumPy arrays and CFFI-compiled CPU kernels.
- `ContextCupy` uses CuPy arrays and CUDA kernels.
- `ContextPyopencl` uses PyOpenCL arrays and OpenCL kernels.

OpenMP support for CPU kernels requires an OpenMP-capable compiler and runtime
to be installed in the environment.


## Development

Run the test suite with:

```bash
pytest tests
```

Some tests and examples need optional GPU dependencies or a configured GPU
runtime. CPU-only development can still run the regular test subset that does
not require those contexts. Which tests are run is specified by the `XOBJECTS_TEST_CONTEXTS` flag. By default all contexts supported by the system are used, but this can be narrowed down, e.g. only the CPU serial and OpenMP tests will be run if `XOBJECTS_TEST_CONTEXTS=ContextCpu;ContextCpu:auto` is set.

## Contributing

The project uses Black for Python formatting and provides a pre-commit
configuration. Install the pre-commit hooks before committing:

```bash
pip install pre-commit
pre-commit install
```

This will run formatting hooks on every commit.


## Contexts and Buffers

A context represents where data lives and where kernels execute. A buffer is a
memory allocation owned by a context. Xobjects can allocate a new buffer
implicitly, or they can be placed explicitly in an existing buffer so that
several objects share one memory range.

```python
import xobjects as xo

ctx = xo.ContextCpu()
buffer = ctx.new_buffer(capacity=1024)

class Point(xo.Struct):
    x = xo.Float64
    y = xo.Float64

p1 = Point(x=1.0, y=2.0, _context=ctx)   # new buffer on ctx
p2 = Point(x=3.0, y=4.0, _buffer=buffer) # explicit buffer

print(p1._context)
print(p2._buffer is buffer)
```

If no `_context` or `_buffer` is supplied, Xobjects uses the default CPU
context. Passing `_context` creates a new buffer in that context. Passing
`_buffer` places the object in that buffer. Passing `_offset` is supported for
advanced use, but it must be used carefully because overlapping allocations can
corrupt objects in the same buffer.

Context arrays can be converted to and from NumPy arrays through the context:

```python
import numpy as np
import xobjects as xo

ctx = xo.ContextCpu()
host = np.arange(8, dtype=np.float64)

on_context = ctx.nparray_to_context_array(host)
back_on_host = ctx.nparray_from_context_array(on_context)
```

On GPU contexts, array attributes expose NumPy-like GPU arrays. Some NumPy
operations require an explicit copy back to host memory with
`nparray_from_context_array`.

`ContextCpu` can compile kernels with OpenMP support when it is created with
`omp_num_threads` different from `0`:

```python
import xobjects as xo

ctx = xo.ContextCpu(omp_num_threads="auto") # let OpenMP choose the thread count
ctx = xo.ContextCpu(omp_num_threads=4)      # use four OpenMP threads
ctx = xo.ContextCpu(omp_num_threads=0)      # serial CPU execution
```

## Defining Data

Xobjects data types describe a stable binary layout. The same definition drives
Python accessors, buffer allocation, serialization helpers, and generated C
accessors.

### Scalars

Scalar types wrap fixed-width NumPy dtypes and C types:

- floating point: `Float64`, `Float32`;
- signed integers: `Int64`, `Int32`, `Int16`, `Int8`;
- unsigned integers: `UInt64`, `UInt32`, `UInt16`, `UInt8`.

```python
import xobjects as xo

value = xo.Float64(3.5)
index = xo.Int64(7)
```

### Arrays

Arrays are declared by indexing an item type:

```python
import xobjects as xo

Vector = xo.Float64[3]   # fixed-length array
Series = xo.Float64[:]   # one dynamic dimension
Matrix = xo.Float64[:, 3]

v = Vector([1.0, 2.0, 3.0])
s = Series([0.0, 1.0, 2.0, 3.0])
m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

Arrays can contain scalars, structs, strings, and references. Fixed-size arrays
have a static footprint; dynamic arrays store size and layout metadata in the
buffer before their data.

### Structs

`Struct` classes define named fields stored together in one xobject.

```python
import xobjects as xo

class Particle(xo.Struct):
    x = xo.Float64
    px = xo.Float64
    state = xo.Int64

particle = Particle(x=1.0, px=0.01, state=1)
particle.x += 0.5
```

Fields can be scalars, arrays, nested structs, strings, `Ref` values, or
`UnionRef` values. Dynamic fields are stored through offsets inside the same
buffer-backed object.

### Hybrid Classes

`HybridClass` is a convenience layer for user-facing Python classes that keep
regular Python methods and attributes while storing selected fields in an
underlying xobject. The C-visible fields are declared in `_xofields`.

```python
import xobjects as xo

class Record(xo.HybridClass):
    _xofields = {
        "value": xo.Float64,
        "samples": xo.Float64[:],
    }

    def add_offset(self, offset):
        self.value += offset

record = Record(value=1.5, samples=[0.0, 1.0, 2.0])
record.add_offset(0.5)

print(record.value)
print(record._xobject.samples)
```

The attributes listed in `_xofields` are backed by the same memory as
`record._xobject`. Array fields are exposed as NumPy or NumPy-like arrays on the
owning context.

### Strings and References

`String` stores a null-terminated variable-length string. `Ref(T)` stores a
reference to another object of type `T` in the same buffer. `UnionRef` stores a
reference that can point to one of several declared xobject types.

References are useful when multiple objects need to share data without copying
it. They must refer to objects in the same buffer because the stored value is a
relative offset.

## The C Programming Model

Every xobject type can generate C declarations and accessor functions for its
buffer layout. This lets kernel code read and write fields without hard-coding
offsets.

```python
import xobjects as xo

class Point(xo.Struct):
    x = xo.Float64
    y = xo.Float64

print(Point._gen_c_api().source)
```

The generated API includes functions such as:

```c
double Point_get_x(const Point obj);
void Point_set_x(Point obj, double value);
double* Point_getp_x(Point obj);
```

For array fields, the generated API also includes length, getter, setter, and
pointer helpers. For references and unions, it includes helpers that recover the
referenced object and type information.

Handwritten C code should use the generated accessors instead of assuming a
particular byte layout. That keeps the source portable across CPU, CUDA, and
OpenCL backends and across static or dynamic field layouts.

Xobjects provides portability macros through `xobjects/headers/common.h`, for
example `GPUFUN`, `GPUKERN`, `GPUGLMEM`, and `RESTRICT`.

## Defining Kernels

A kernel is a compiled function registered on a context. Its source is C-like
code and its Python-side signature is described with `xo.Kernel` and `xo.Arg`.
Xobjects uses the signature to generate the required type API, compile the
source for the selected backend, and expose the result through `context.kernels`.

```python
import numpy as np
import xobjects as xo

ctx = xo.ContextCpu()

source = r"""
GPUKERN
void scale_and_add(const int32_t n,
                   const double a,
                   GPUGLMEM const double* x,
                   GPUGLMEM double* y)
{
    VECTORIZE_OVER(i, n)
        y[i] = a * x[i] + y[i];
    END_VECTORIZE
}
"""

kernels = {
    "scale_and_add": xo.Kernel(
        args=[
            xo.Arg(xo.Int32, name="n"),
            xo.Arg(xo.Float64, name="a"),
            xo.Arg(xo.Float64, pointer=True, const=True, name="x"),
            xo.Arg(xo.Float64, pointer=True, name="y"),
        ],
        n_threads="n",
    )
}

ctx.add_kernels(
    sources=[source],
    kernels=kernels,
    extra_headers=['#include "xobjects/headers/common.h"'],
)

x = ctx.nparray_to_context_array(np.arange(5, dtype=np.float64))
y = ctx.nparray_to_context_array(np.ones(5, dtype=np.float64))

ctx.kernels.scale_and_add(n=len(x), a=2.0, x=x, y=y)
print(ctx.nparray_from_context_array(y))
```

Kernels can also take xobject structs or arrays as arguments:

```python
import xobjects as xo

class Point(xo.Struct):
    x = xo.Float64
    y = xo.Float64

source = r"""
GPUFUN
double norm2(Point p)
{
    return Point_get_x(p) * Point_get_x(p)
         + Point_get_y(p) * Point_get_y(p);
}
"""

kernels = {
    "norm2": xo.Kernel(
        args=[xo.Arg(Point, name="p")],
        ret=xo.Arg(xo.Float64),
    )
}

ctx = xo.ContextCpu()
ctx.add_kernels(
    sources=[source],
    kernels=kernels,
    extra_headers=['#include "xobjects/headers/common.h"'],
)

point = Point(x=3.0, y=4.0, _context=ctx)
print(ctx.kernels.norm2(p=point))
```

For reusable classes, kernel descriptions can be attached to a `Struct` or
`HybridClass` through `_kernels` and compiled with `compile_kernels()`.

Loops written with `VECTORIZE_OVER` use the execution policy of the selected
context. On a serial CPU context they become regular `for` loops. On an OpenMP
CPU context they become OpenMP parallel loops. On GPU contexts they map to the
backend thread index.

## Moving and Copying

Hybrid objects support moving or copying data across buffers and contexts:

```python
import xobjects as xo

class Item(xo.HybridClass):
    _xofields = {
        "value": xo.Float64,
    }

cpu_item = Item(value=1.0, _context=xo.ContextCpu())

other_cpu_ctx = xo.ContextCpu()
other_item = cpu_item.copy(_context=other_cpu_ctx)
cpu_item.move(_buffer=xo.context_default.new_buffer())
```

Objects with references have stricter rules because references are buffer-local.
When using `Ref` or `UnionRef`, keep the referencing object and the referenced
object in the same buffer.

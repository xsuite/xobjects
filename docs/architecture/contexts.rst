Contexts
========

Devices
-------


Buffers
-------

Buffers are able to store data in the host or device from several object types and return data in device or host.


Instance methods:

*  write(self, source, offset, nbytes): write nbytes from source that can be a np-like array, bytearray, bytes or another buffer
*  to_nplike(self,dtype,shape,offset): return a view of the data in a np-like arrays
*  to_bytearray(self,offset,count): return

.. code::

   def update_from_xbuffer(self,offset, source, source_offset, nbytes):
       "update from any xbuffer, don't pass through gpu if possible"

   def update_from_buffer(self, offset, buffer):
       """source is python buffer such as bytearray, numpy array .data"""

   def copy_native(self,offset,nbytes):
        """return native data with content at from offset and nbytes
        return native_data

   def update_from_native(self, offset, source, source_offset, nbytes):
       """update native source into buffer usef in update_from_xbuffer"""

   def to_nplike(self,offset,shape,dtype)
       """view in nplike"""

   def to_bytearray(self,offset,nbytes)
       """copy in byte array: used in update_from_xbuffer"""

   def to_pointer_arg(self,offset,nbytes):
       return self.data[offset:offset+nbytes] #offset can lead to alignement error in opencl if not multiple of 4 bytes



Arguments for kernels:

- CPU:
  - numpy array ->  ``ffi.cast("double *", ffi.from_buffer(arr))``
  - numpy scalar -> no change
  - s string -> ``ss.decode("utf8")``

- PyOpenCL:
  - pyopencl.Buffer: [offset:offset+nbytes] or copy back and fort if needs to be aligned
  - pyopencl.array:  arr.base_data[arr.offset:arr.offset+arr.nbytes] or copy back and fort

- cuda:
    -  pycuda.array: arr or arr.data


nparray.data:    python buffer pointing to start of the array arrays
clarray.data:  Optional Opencl buffer pointing to the start of the array
clarray.base_data,clarray.base_data:  Optional Opencl buffer  and offset from where the pointer start
cupyaray.


clarray.base_data[]:

Code generation
---------------

Each context is able to compile source code and make available python callable functions.

The callable functions can take as arguments Python numbers, strings,
numpy-like arrays (depending on the context) and xobject allocated in the same
contexts (if not copies will be done).

Source code is optionally pre-processed to adapt it to the context.

The callable functions are defined by:
*  the name of the function in the source code
*  the type of parameters
*  how to pass parallalization instruction (threads for openmp, grid for GPU)

The callable functions are parametrized by name and argument types.

Each xobject type can provide the API source code  the ``_get_c_api()`` class method.



C Api
^^^^^^^^^^^^^^^^^^^

General:

- ``<pointer type> <typename>_getp_<field_names>(<typename> obj, <indexes>)``
- ``<scalar type>  <typename>_get_<field_names>(<typename> obj, <indexes>)``
- ``<typename>_set_<field_names>(<typename> obj, <indexes>, <scalar_type> value)``
- ``<int64> <typename>_size_<field_names>(<typename> obj,<indexes>)``

Array:
- ``<int64> <typename>_len...(<typename> obj) : number of itemns``
- ``<int64> <typename>_dim...(<typename> obj, int index): size of dimension``
- ``<int64> <typename>_ndim...(<typename> obj): number of dimensions``
- ``void <typename>_strides(int* result)``

UnionRef:
- ``<int64> <typename>_typeid...(<typename> obj) : get member
- ``void* <typename>_member...(int* result)``


Kernels and Functions
^^^^^^^^^^^^^^^^^^^^^

Context can execute:

- kernels: no return function and executed on a grid
- functions: only for cpu kernels


.. code:: python
  Kernel(
    cname,
    args=Arg(dtype, name: String, const: Boolean)
    grid=integer, string or list of string or functions of arguments
    ret=None)


Arg:

* atype: argument type:
  - provide validation function: cls._validate(value) or atype(value)
  - provide C-type name: atype._get_ctype()  or dtype -> converted to c types,
  - generate raw argument from value .atype._get_carg() for the interface
* name: optional, String needed for the grid
* const: boolean (used in case of copy back)

Code:

* define sources, list of string or files
* define kernels to expose, list of kernels
* define preprocessors
* attach callable kernels to context
* to be used by CMethod or CProperty

.. code::
    double x -> Arg(dtype=xo.Float64,name='x'): takes python scalar and np scalar
    double *restrict x -> Arg(dtype=xo.Float64,pointer=True,const=False,name='x'): takes np.array, xo.Float64 arrays
    const double *restrict x -> Arg(dtype=xo.Float64,pointer=True,const=True,name='x'): takes np.array, xo.Float64 arrays
    const Float64_N -> Arg(dtype=xo.Float64[:],const=True,name='x'): takes xo.Float64[:]
    const Float64_6by6 -> Arg(dtype=xo.Float64[6,6],const=True,name='x'): takes xo.Float64[6,6]





*  function: general function
*  method: implies first attribute is the instance of the class where it is defined
*  kernel: implies some iteration and no return type

The function can be defined using:
*   header
*   source
*   body

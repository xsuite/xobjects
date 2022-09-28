# Architecture

## Introduction

The library is based on:
- Buffers: manage chunck of bytes on CPU and GPU (Cuda and OpenCL) memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
- Basic types: a set of basic types that allow to describe structured data. Basic types are scalars, strings, structs, arrays and unions. User defined classes, composed with basic types, allow to interact with data on buffers and generate C functions to operate with data.
- C-API generation: The type information allows to write C functions that can be interact with data. The C functions are specialized for CPU, OpenCL and Cuda. Also loops described by annotation in comments are automatically generated for CPU, OpenCL and Cuda.

When a user defined class is created, Python methods and additional metadeta is attached to the class.
When an instance of a user defined class is instantiated, memory is written to a buffer and a Python object is created to interact with the data.
The size of an instance may not be known at class defintion time, but it cannot be changed after initialization.

## Contexts and Buffers

A context contain:

- `new_buffer(capacity)`: create new buffer of initial capacity
- `buffers`: list of weakref to buffers
-  kernels : Object containing compiled code from sources/

Buffer:
- `buffer`: low level buffer
- `allocate(self,size)`: return the offset of the first free memory chunk of given size and grows the buffer if necessary
- `free(self,offset,size)`: free a chuck of memory of a given size
- `grow(self,capacity)`: increase the size of a buffer
- `_new_buffer(capacity)`: creae lowlevel buffer
- `update_from_native(...)`: Copy data from native buffer into self.buffer starting from offset
- `update_from_buffer(...)`: Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data
- `update_from_nplike(...)`: Copy data from nplike matching dest_dtype
- `update_from_xbuffer(...)`: Copy data from any xbuffer, don't pass through cpu if possible
- `copy_to_native(...)`: Copy data from self.buffer into dest
- `to_native(...)`: Return native data with content at from offset and nbytes
- `to_nplike(...)`: Return a view in of an nplike
- `to_bytearray(...)`: Return a byte array: used in update_from_xbuffer
- `to_pointer_arg(...)`: Return data that can be used as argument in kernel

## Types

Types can be composed of:
- scalar: numbers, String
- compound: Struct, Array, Ref, UnionRef 

### Scalars
- examples: Float64, Int64, ...
- create: Float64(3.14)
- memory layout
    - data 

### String:
- create: String(string_or_int)
- memory layout
    - size
    - data 


### Struct
- example: struct MyStruct field1 field2 ...
- create: Struct(dict), Struct(**args)
- memory layout:
  - [ instance size ]
  - static-field1
  -  ..
  - static-fieldn
  - [ offset field 2 ]
  - [ offset ...     ]
  - [ offset field n ]
  - [ dynamic-field1 ]
  - [ ... ]
  - [ dynamic-fieldn ]

### Array:
- example: array f64 d1 d2 d3 'C' ; array i64 d1 : d3 'F' ;
- create: Array(d1,d2,...) or Array([...]) or Array(np-array)
- memory layout:
  - [size] if not _is_static_shape or not _is_static_type
  - [dims ... ] len(_dynamic_shape)
  - [strides...] if nd>1 and dynamic shapes
  - [offsets] if not _is_static_type
  - data

### Union:
- create: Union(obj), Union( (typename,args) )
- memory layout
  - typeid
  - data

### UnionRef:
- create: UnionRef(obj), UnionRef( (typename,args) )
- memory layout
  - typeid
  - offset


### Ref:
- memory layout
  - offset

## Implementation details

Assumptions:
- Python user API cannot change the internal structure of the object but only the content
- Python object caches the structure information to avoid round trips to access data
- C user API does not manipulate structures


Operation:

- _inspect_args(*args, **args) -> info: Info
  - check value and calculate size, offsets and dimensions from arguments, recursively
  - returns at least `size` for `_get_a_buffer` and `value` for `_to_buffer`

- _get_a_buffer(size, _context, _buffer, _offset):
  - make sure a valid buffer and offset is created

- _to_buffer(buffer, offset, value, info, size?):
  - set data on buffer with offset from pyhon value, using info and implicitely respecting size
  - if value same class -> binary copy (passing from context if needed)

- _from_buffer(buffer, offset) -> value:
  - create python object from data on buffer. Can be a native type (scalar or string) or another XObject (struct, array)


- __init__(self,*args,_context=None, _buffer=None, _offset=None,*nargs)
  - initialize object data on buffer from args and create python object:
    - use _pre_init(*args,**nargs) -> return args, nargs standard initialization value
    - check values and calculate size using `_inspect_args`, return info object containing size, offsets and values to be later used by _to_buffer
    - allocate using `_get_a_buffer`
    - write data using `_to_buffer`
    - use `_post_init` to complete object
`
- _update(self, value, size=None):
  - Optional update object content using value, in case respecting size for string

- __get__(field,instance) or __getitem__(self,index...)
  - if return instance._cache[field.index] else  #should implement item caching for struct and array
  - else return _from_buffer

- __set__(..., value) or __setitem__(...,value)
  - if hasattr(self.ftype._update) get object and update in place
    else call _to_buffer

- json enconding
  nested object can be initialized using a json enconding
  d:dict -> type(**d)
  t:tuple -> type(*t)
  a:other -> type(a)



Class content:

- _inspect_args(cls, *args, **nargs) -> size, offsets
  compute size and sizes and offsets from arguments

- _to_buffer(cls, buffer, offset, value, offsets=None):
  initialize object on buffer

- _from_buffer(cls, buffer,offset):
  create python object from initlialized object on buffer,offset

- _size: object size
  None if dynamic

Instance content:

- _get_size(self) -> size

## C-API

Access
- `<Class>_get<_attrs>(obj, <indexes>)` -> scalar or object pointer
- `<Class>_set<_attrs>(obj, <indexes>, scalar)`

e.g. Multipole_get_field_normal(3)

Kernel convention:

```
<fun>(void **refs){
          Type0 arg0 = (Type0) refs[0];
          ....
          Type1 argN = (TypeN) refs[N];
        ...
        }
```

``` <fun>(buffer1, offset, ..., offset3){
         Type0 arg0 = (Type0) refs[offset0];
         ,,,
         TypeN argN = (TypeN) refs[offsetN];
}
```

## TODO

- add free methods to delete data from buffer
- implement SOA types

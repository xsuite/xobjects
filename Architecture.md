# Architecture

## Introduction

The librarties is based on:
- Buffers: manage dynamic unstructured data on CPU and GPU memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
- Basic types: a set of basic types that allow to describe structured data. Basic types are scalars, strings, structs, arrays and unions. User defined classes, composed by basic types, allow to interact with data on buffers and generate C functions to operate with data.
- Api: type system describing structured data. Types are composed using scalars, string, struct, array, union primitives. Recursive types are forbidden.
- Types generate python objects that can create and modify objects on buffers and generate C operate with data.


## Contexts and Buffers

A context contain:

- `new_buffer(capacity)`: create new buffer of initial capacity
- `buffers`: list of weakref to buffers
- contain kernels...

Buffer:
- `allocate(self,size)`
- `free(self,offset,size)`
- `grow(self,capacity)`
- `buffer`: low level buffer
- `_new_buffer`: lowlevel buffer

Buffer review:



1) write_from_array: any python object exposing the buffer protocol
    - numpy array
    - __array_interface__
    - buffer_interface
    - cupy array (similar use case for copy from samebuffer)
    - opencl array (similar use case for
2) copy_from_xbuffer: any Buffer
3) copy_from_samebuffer: Buffer from the same context

## Types

Types can be composed of:
- scalar: numbers, string
- compound: struct, array, ref

### Scalars
- examples: Float64, Int64, ...
- create: Float64(3.14)

### String:
- create: String(string_or_int)
- layout
  - size
  - bytes

### Struct
- example: struct  MyStruct field1 field2 ...
- create: Struct(dict), Struct(**args)
- layout:
  [ instance size ]
  static-field1
  ..
  static-fieldn
  [ offset field 2 ]
  [ offset ...
  [ offset field n ]
  [ dynamic-field1 ]
  [ ... ]
  [ dynamic-fieldn ]

### Array:
- create: Array(d1,d2,...) or Array([...]) or Array(nparray)
- example: array d1 d2 d3 'C' ; array d1 : d3 'F' ;
- layout:
  - [size] if not _is_static_shape or not _is_static_type
  - [dims ... ] len(_dynamic_shape)
  - [strides...] if nd>1 and dynamic shapes
  - [offsets]
  - data

### Union:
- create: Union(element), Union( (typename,{}) )
- layout
  - typeid
  - offset [or data and introducing Ref?]

### Ref?:
- layout
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
    - use _pre_init(*args,**nargs) -> return args, nargs tandard initialization value
    - check value and calculate size using _inspect_args
    - get resources using _get_a_buffer
    - write data using _to_buffer
    - set python object cached data

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

- debug C getter and setter
- implement Ref C setter and getter

- add __del__ methods for Struct, Array, Ref
- implement SOA

### Later

- Make types read-only instances to avoid unsafe run-time operations
- Consider exposing XBuffer and moving specialization to XContext
- Consider Buffer[offset] to creating XView and avoid _offset in type API
- Consider mutable string class
- Consider scalar classes to allows subclassing default and init for data validation this would avoid using field in structs
- Use __slots__ where possible

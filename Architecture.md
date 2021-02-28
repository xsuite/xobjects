# Architecture

## Introduction

The librarties is based on:
- Buffers: manage dynamic unstructured data on CPU and GPU memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
- Api: type system describing structured data. Types are composed using
  scalars, string, struct, array, union primitives. Types generate python objects that can create and modify objects on buffers and generate C operate with data.


## Contexts and Buffers

Context:

- new_buffer(capacity): create new buffer of initial capacity
- buffers: list of weakref to buffers
- <run kernels>

Buffer:
- allocate(self,size)
- free(self,offset,size)
- grow(self,capacity)

- .buffer: low level buffer
- read(offset,size) -> bytearray
- write(offset,size,data):

- _new_buffer: lowlevel buffer
- copy_to(self, dest: bytearray)
- copy_from(self,source, src_offset, dest_offset, byte_count): source must be of the same time


## Types

Types can be composed of:
- scalar: numbers, string
- compound: struct, array, list, union

The scalar types are returned as copy, while the compound type are views of the memory behind.


Types can be
- static-size: scalar, struct of static-size members, fixed dimension array of static-size member, union of flexible-size members, reference
- flexible-size: does not change after initialization, string, arrays with flexible-size members
- dynamic-size: can change after creation


### Struct
struct name fname1 ftype1 ... fnameN ftypeN
array mtype dim1 ... dimN 


## Data layout is part of the specification.

Struct:
  [ instance size ]
  static-field1
  ..
  static-fieldn
  [ offset field 2 ]
  [ offset ...
  [ offset field n ]
  [ dynamic-field1 ]
  [ ...
  [ dynamic-fieldn ]

Array:
  [size]
  [ndims]
  [dims ... ]
  data


## Implementation details

Initialization option:
- _context, _buffer, _offset can be given to specify resources

init:
   struct() or struct(**dict) or struct(dict) or struct(another_struct)
   string(string_or_length)
   array(dimensions as integers or iterable)
   union(value) or union(_type=...,**fields)

Initialization steps:
  - compute size from args: if no resources
  - acquire resources or using resources
  - set args

Field attributes
  - type (with or without internal constraints?)
  - read-only
  - property
  - default or factory
  - initialization only

generic api class:
  _size: None if dynamic
  _to_buffer(cls,buffer, offset, value):
      - store a python object into buffer
      - used by __set__ in Struct
  _from_buffer(cls,buffer,offset):
      - create a python object from buffer, offset
      - used by __get__ in Struct
  __init__(self,_context, _buffer, _offset, *args, **nargs):
         - allocate object in buffer from BytearrayContext, or _context or _buffer
         - use arguments to populate data
         - Struct(**fields)
         - String(int_or_string)
         - Array(dim1_or_iterable, dim2, ..., dimn)
         - Union(_type=...,**fields)
  _get_size(self): Instance size
  _get_size_from_args(cls,*args,**nargs) -> size, extra:
         - compute size from arguments
         - mimic __init__ signature cannot use tuple or dict as arguments
  _parse_args(self,*args,**args): use to customize class creation

struct api class
  _d_offsets (instance): offsets for dynamic fields
  _size (class or instance): struct_size


   For nested types:
   StructA(field0 = a, field1 = (a,b,c), field2 = {'a':a , 'b': b} )  ->
   {field0 = field0.ftype(a),
    field1 = field1.ftype(a,b,c),
    field2 = field2.ftype(a=a,b=a}


## C-APi

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

- Consider return opencl array rather than bytearray
- Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
- Consider creating an `api` object type factory
- Consider Buffer[offset] to create View and avoid _offset in type API
- Consider mutable string class
- Consider to add arg_default (in alternative of factory)
- Add to_buffer user in context memory

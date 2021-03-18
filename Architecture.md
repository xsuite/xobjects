# Architecture

## Introduction

The librarties is based on:
- Buffers: manage dynamic unstructured data on CPU and GPU memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
- Basic types: a set of basic types that allow to describe structured data. Basic types are scalars, strings, structs, arrays and unions. User defined classes, composed by basic types, allow to interact with data on buffers and generate C functions to operate with data.
- Api: type system describing structured data. Types are composed using scalars, string, struct, array, union primitives.
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
- `read(offset,size)` -> bytearray
- `write(offset,size,data)`:

- `_new_buffer`: lowlevel buffer
- `copy_to(self, dest: bytearray)`
- `copy_from(self,source, src_offset, dest_offset, byte_count)`: source must be of the same time


## Types

Types can be composed of:
- scalar: numbers, string
- compound: struct, array, list, union

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
  - [size]
  - [dims ... ]
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

- _get_a_buffer(size, _context, _buffer, _offset):
  - make sure a valid buffer and offset is created

- _from_buffer(buffer, offset) -> value:
  - create python object from data on buffer. Can be a native type (scalar or string) or another XObject (struct, array)

- _to_buffer(buffer, offset, value, info, size?):
  - set data on buffer with offset from pyhon value, using info and implicitely respecting size
  - if value same class -> binary copy (passing from context if needed)

- __init__(self,_context, _buffer, _offset, ...)
  - initialize object data on buffer from args and create python object
  - check value and calculate size using _inspect_args
  - get resources using _get_a_buffer
  - write data using _to_buffer
  - set python object

- _update(self, value, size=None):
  - Optional update object using value, in case respecting size for string

- __get__(field,instance) or __getitem__(self,index...)
  - if return instance._cache[field.index] else  #should implement item caching for struct and array
  - else return _from_buffer

- __set__(..., value) or __setitem__(...,value)
  - if hasattr(self.ftype._update) get object and update in place
    else call _to_buffer


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

### Critical
- add check shape from argument
- implement Array
- implement Union
- test type creation then type access
- review struct tests due to dynamic caching offsets
- speed-up array creation for basic types

### Later

- Consider return opencl array rather than bytearray
- Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
- Consider creating an `api` object type factory
- Consider Buffer[offset] to create View and avoid _offset in type API
- Consider mutable string class
- Add String.to_buffer to use same context copy
- Make types read-only instances to avoid messing
- Evaluate round-trips versus caching

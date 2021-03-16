# Architecture

## Introduction

The librarties is based on:
- Buffers: manage dynamic unstructured data on CPU and GPU memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
- Basic types: a set of basic types that allow to describe structured data. Basic types are scalars, strings, structs, arrays and unions. User defined classes, composed by basic types, allow to interact with data on buffers and generate C functions to operate with data.


## Contexts and Buffers

Context:

- new_buffer(capacity): create new buffer of initial capacity
- buffers: list of weakref to buffers
- run kernels

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

###Scalars
examples: Float64, Int64, ...
create: Float64(3.14)

###String:
create: String(string_or_int)
layout
  size
  bytes

###Struct
example: struct  MyStruct field1 f ... 
create: Struct(dict), Struct(**args)
layout
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

###Array:
create: Array(d1,d2,d3) or Array([...]) or Array(nparray)
layout
  [size]
  [ndims]
  [dims ... ]
  [offsets]
  data

###Union:
create: Union(element), Union( (typename,{}) )

layout
  typeid
  offset [or data?]


## Implementation details

| calculate resources for object on buffer from args | _get_size_from_args |
| acquire resources for object on buffer from size |  _get_a_buffer |
| create python object from initialized buffer |  _from_buffer |

| initialize buffer from value  | _to_buffer  |
| create python object from args | __init__ |


initialize buffer from value:
  - needs offsets <- get_size_from_args
  - has buffer offset but cannot check size in buffer [not safe]
  - write offsets
  - write values (and pass offsets)

create python object from args:
  - needs size and offsets <- get_size_from_args
  - needs buffer offsets <- _get_a_buffer
  - write offsets
  - write values (and pass offsets)
  - need to create python object








Generic:

- _get_a_buffer(size,context,buffer,offset) -> buffer,offset
  get a valid buffer and offsets, creating buffer and allocating space if necessary

Class content:

- _get_size_from_args(cls, *args, **nargs) -> size, offsets
  compute size and sizes and offsets from arguments

- _to_buffer(cls, buffer, offset, value, offsets=None):
  initialize object on buffer

- _from_buffer(cls, buffer,offset):
  create python object from initlialized object on buffer,offset

- _size: object size
  None if dynamic

Instance content:

- _get_size(self) -> size

Field class:
- ftype
- index
- name
- offset
- deferenced
- readonly

Struct class:
- _fields: list of fields
- _d_fields: list of dynamic fields
- _s_fields: list of dynamic fields

Struct instance:
- _offsets: cached offsets of dynamic fields dict indexed by field.index
- _sizes: cached sizes of dynamic fields dict indexed by field.index





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

# Architecture

## Introduction

The librarties is based on:
- Buffers: manage dynamic unstructured data on CPU and GPU memory using different backends
- Contexts: keep track of buffers, launch functions operating on buffers
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

Types can be
- static-size: scalar, struct of static-size members, fixed dimension array of static-size member, union of flexible-size members, reference
- flexible-size: does not change after initialization, string, arrays with flexible-size members
- dynamic-size: can change after creation



### String
Definition: String [size]
Python Initialization: StructA(int_or_string)
JSON: "..."

### Struct
Definition: struct name fname1 ftype1 ... fnameN ftypeN
Python Initialization: StructA(**fields), StructA(a_dict), StructA(another_StructA)
JSON: {...}

### Array
Definition: array mtype dim1 ... dimN
Dimensions can be integer or None
Initialization: ArrayA(iterable) compatibile dimension or StructA(d1,d2,d3) unknown dimension
JSON: [...]

### Union
Definition: union type1 type2 ... typeN
Python Initialization: UnionA(element), UnionA(typename,arg1,...,argn)
JSON: (type,arg1,...)

Initialization from JSON for compound type:
- if arg is tuple -> *arg
- elif arg is dict -> **arg
- else arg -> arg

The scalar types are returned as copy, while the compound type are views of the memory behind.


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

Union:
  [size]
  typeid
  data

## Implementation details

Serialization tasks:
- _get_size_from_args(*args, **nargs)  [specialized]
  Returns the size and the full offsets data of the hierarchy
   -> size: int , offsets: None or dict (None is the root, 'name' or index are the leaf
  - used in __init__
  - init:
    - string(string_or_int)
    - struct(**nargs), struct(adict), struct(a_struct)
    - array(d1,d2,...), array(iterable), array(an_array)


- _get_a_buffer(size,context,buffer,offset)  [generic]
   -> context,buffer,offset, size: offsets:list[pairs], extra: dict[part,offset]
   - make sure memory region is available
   - create or reuse a context  -> a context exists
   - create (need size) or reuse a buffer  -> a buffer exists
   - create (need size) or use a buffer region  -> an offset exists
   - used in _init_

- _write_offsets(buffer,offset, offsets, extra) [generic]
- write offsets if needed into buffer, offset
  - need offsets unless class does not need or value is an xobject
  - done in _init_ recursively


- _to_buffer(buffer,offset, value): [specialized]
  - can be used only if memory is prepared
  - if value is xobject:
     _copy(buffer,offset, xobject)
  - else
     _value_to_buffer(buffer,offset, value)

- _copy(buffer,offset, xobject)
 - check if size is correct (optionally?)
 - copy memory in xobject to buffer

- _check_value(buffer,offset, value)
  - check if value is compatible with type
  - used by __set__ and __setitem__


- read data from buffer, offset
  store buffer, offset
  cache offsets?


Initialization option:
- _context, _buffer, _offset can be given to specify resources

init:
   struct() or struct(**dict) or struct(dict) or struct(another_struct)
   string(string_or_length)
   array(dimensions as integers or iterable)
   union(value) or union(_type=...,**fields)

Object creation:
  - compute size and offsets from args
  - acquire resources or use resources
  - set offsets
  - set args

Object reference:
  - set buffer and offset

Field attributes
  - type (with or without internal constraints?)
  - read-only
  - property
  - default or factory
  - initialization only

generic api class:
  _size: None if dynamic
  _to_buffer(cls, buffer, offset, value):
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
- Consider to add arg_default (in alternative of factory)
- Add String.to_buffer to use same context copy
- Make types read-only instances to avoid messing
- Evaluate round-trips versus caching

Types
=====

Introduction
-------------


Data layout and API can be defined by a composition of:
*   Scalars: Int8, Int16, Int32, Int64, Float32, Float64
*   Compounds: Struct, Array, Ref

Types can have a fixed size (*static types*) or a variable size (*dynamic size*).
The size of one instance cannot change after creation, but their content can.


Data Layout
-------------

Data structure is organized in 64bit slots.

Table indicates slot size and the description.

Scalar

=====  ===========
Slots  Description
=====  ===========
8      Data
=====  ===========

String of n-bytes

=============     ===============================
Slots             Description
=============     ===============================
8                 Size in bytes
ceil((n+1)/8)     Data in UTF-8 null terminated
=============     ===============================


Struct with static and *n* dynamical fields

=======  ===============================
Slots    Description
=======  ===============================
8        Optional: size if n>0
...      Fixed-size field data
8*(n-1)  Optional: Offsets of the 2nd to n-th last variable sized
...      Optional: Variable sized data
=======  ===============================

Array with n dimensions, whose l are dynamic of *m* items.

=======  ===============================
Slots    Description
=======  ===============================
8        Optional: size
8*l      Optional: size of l-variable dimensions
8*n      Optional: strides for n-dimensions larger than 2
m*s      Data or offsets for m-items (s is the item-size or 8)
...      Optional: Items data if the item type is dynamic
=======  ===============================

Union

=======  ===============================
Slots    Description
=======  ===============================
8        Type id
...      Item data
=======  ===============================


Ref

=======  ======================================
Slots    Description
=======  ======================================
8        Offset from start of the buffer
8        Optional: typeid
=======  ======================================


Python type interface
----------------------

The following describe the generic interface

Class variables:

==============  ===============================
name            Description
==============  ===============================
_size           Size in bytes, None if not static size
==============  ===============================

Class methods:

============= ============== ============ ============================================
name          args           return                  Description
============= ============== ============ ============================================
_inspect_args *args,         Info         Return at least info.size and optionally
              *kwargs                     other metadata to build objects in memory
                                          from python objects. Args are propocessed
                                          by _pre_init
_pre_init     *args,         args,        Preprocess arguments and generate standard
              *kwargs        kwargs       values
_to_buffer    buffer         None         Serialize python in buffer from offset.
              offset                      Assume sufficient size is available
              value
_from_buffer  buffer         instance     Return Python instance from buffer and
              offset                      and offset
_to_schema                   string       TODO String representation of the type to
                                          deserialize objects
============= ============== ============ ============================================


Instance variables:

=============  ========================================
name           Description
=============  ========================================
_size          Optional: size in bytes for dynamic-size
=============  ========================================


Class methods:

============= ============== ============ ============================================
name          args           return                  Description
============= ============== ============ ============================================
_post_init                                Run after object creation in __init__
copy          buffer or      None         TODO Return a copy into a buffer
              context                      Assume sufficient size is available
_update       value          instance     Update values of an exsiting objec
_to_json                     json         TODO json
============= ============== ============ ============================================



**__init__**

init is used to  create a new xobject by allocating memory on the buffer.
Ref and Scalars cannot be initialized alone (maybe temporarily)

Init can take by default:

*  a json object:
    *  dict for Struct
    *  list for Array,
    *  number of Scalars
    *  String for String
    *  (Typename, {}) for Ref of multiple types
*  dimensions (for string or arrays)
*  an xobject of the correct type
*  None for Ref

The data is dispatched to _pre_init using the following convetion:
* a dict is passed with **
* a tuple is passed with *
* any other object is passed verbatim

Init uses the following steps:

*  pre_init to pre_process input
*  inspect_args to calculate sizes and collect metadata
*  get_a_buffer to allocate space
*  _to_buffer to write to memory
*  _from_buffer to build a python object
*  post_init

**_from_buffer**

When accessing data inside compound object the _from_buffer method is used.
For scalar and string, python data is returned. For coumpund object,
XObjects are returned

**setters**
Setters uses to_buffer or _update to update existing data.

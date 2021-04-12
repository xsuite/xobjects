Types
=====

Data Layout
-------------


Data layout and API can be defined by a composition of:
*   Scalars: Int8, Int16, Int32, Int64, Float32, Float64
*   Compounds: Struct, Array, Union, UnionRef, SOA, Ref

Types can have a fixed size (*static types*) or a variable size (*dynamic size*).
The size of one instance cannot change, but their content can.



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
=======  ======================================


Python type interface
----------------------

Generic interface
^^^^^^^^^^^^^^^^^^


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
                                          by _pre_init                                          _pre_init
_pre_init     *args,         args,        Preprocess arguments
              *kwargs        kwargs
_to_buffer    buffer         None         Serialize python in buffer from offset.
              offset                      Assume sufficient size is available
              value
_from_buffer  buffer         instance     Return Python instance from buffer and
              offset                      and offset
_to_schema                   string       String representation of the type to
                                          deserialize objects
============= ============== ============ ============================================


Instance variables:

=============  ========================================
name           Description
=============  ========================================
_size          Optional: size in bytes for dynamic-size
=============  ========================================

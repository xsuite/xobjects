# XObjects

Library to create and manipulate serialized object in CPU and GPU memory efficiently.

The library defines:
-  Contexts: create buffers and manage computations e.g. BytearrayContext, CLContext, ...
-  Buffers: reference to internal memory buffer with an allocator e.g.
   BytearrayContext, CLBuffer, ...
-  Data types: define users define types to create and manipulate objects on
   buffers and generate C api, e.g. Float64, Int64, String, Struct, Array, Union

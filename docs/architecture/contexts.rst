Contexts
========

Devices
-------


Buffers
-------


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

Each xobject type can provide the API source code  the `_get_capi()` class method.




Setters and getters
^^^^^^^^^^^^^^^^^^^

Functions
^^^^^^^^^

*  function: general function
*  method: implies first attribute is the instance of the class where it is defined
*  kernel: implies some iteration and no return type

The function can be defined using:
*   header
*   source
*   body

.. XObjects documentation master file, created by
   sphinx-quickstart on Thu Apr  8 23:39:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XObjects's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


XObjects is a library to create and manipulate serialized object in CPU and GPU memory efficiently in Python and C.

The library defines:
*  Contexts: create buffers and manage computations e.g. BytearrayContext, CLContext, ...
*  Buffers: reference to internal memory buffer with an allocator e.g.
*  BytearrayContext, CLBuffer, ...
*  Data types: define users define types to create and manipulate objects on
buffers and generate C api, e.g. Float64, Int64, String, Struct, Array, Union



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

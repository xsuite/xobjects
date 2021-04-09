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

Example::

    import xobjects as xo
    class Point(xo.Struct):
         x=xo.Float64
         y=xo.Float64
         z=xo.Float64

    class Mesh(xo.Struct):
         points = Point[:]
         edges = xo.int64[:]

   ctx= xo.OpenclContext(device="0.0")
   mesh = Mesh(points=10,edges=10, _context=ctx)

Content
############

* `Quickstart`_
* `Reference`_
* `Architecture`_



Indices and tables
###################


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

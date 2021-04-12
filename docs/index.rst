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

Example (tentative)::

    import xobjects as xo
    import numpy as np

    class Point(xo.Struct):
         x=xo.Float64
         y=xo.Float64
         z=xo.Float64
         hello = xo.CFunction(
         _include=["<stdio.h>"],
         _src ='void hello(){ printf("Hello!\n" );}' )
         dist = xo.CMethod(
            a=Point, b=Point,
            _include=["<math.h>"],
            _body="""
            double dx=Point_get_x(b)- Point_get_x(a);
            double dy=Point_get_y(b)- Point_get_y(a);
            double dz=Point_get_z(b)- Point_get_z(a);
            return sqrt(dx*dx+dy*dy+dz*dz);""")

    class Polygon(xo.Struct):
         point = Point[:]
         edge = xo.Int64[:,2]
         path_length = xo.CMethod(
             poly=Polygon,
             _body="""
             double length=0;
             for (int ii; ii<Polygon_len_edge(poly); ii++){
                aa=Polygon_get_edge(ii,0);
                bb=Polygon_get_edge(ii,1);
                length+=dist(Polygon_get_point(aa),Polygon_get_point(bb));
             };
             return length;""")

   ctx= xo.OpenclContext(device="0.0")
   mesh = Mesh(points=10,edges=10, _context=ctx)
   mesh.points.x=np.random.rand(10);
   mesh.points.y=np.random.rand(10);
   mesh.points.z=np.random.rand(10);
   mesh.points.edges=np.c_[np.arange(10),np.roll(np.arange(10),1)]
   mesh.points[0].hello()
   print(mesh.path_length())

Content
-----------

.. toctree::
   :maxdepth: 3

   quickstart
   reference
   architecture



Indices and tables
--------------------


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

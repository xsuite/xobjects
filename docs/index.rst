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

Example (tentative)

.. code-block:: python

  import xobjects as xo
  import numpy as np

  class Point(xo.Struct):
         x=xo.Float64
         y=xo.Float64
         z=xo.Float64

  class Polygon(xo.Struct):
         point = Point[:]
         edge = xo.Int64[:,2]
         get_length = xo.Method(
            source="""
            #include <math.h>

            double dist(Point a, Point b){
              double dx=Point_get_x(b)- Point_get_x(a);
              double dy=Point_get_y(b)- Point_get_y(a);
              double dz=Point_get_z(b)- Point_get_z(a);
              return sqrt(dx*dx+dy*dy+dz*dz);
            }

            double get_length(Polygon self){
              double length=0;
              for (int ii; ii<Polygon_len_edge(self); ii++){
                aa=Polygon_get_edge(ii,0);
                bb=Polygon_get_edge(ii,1);
                length+=dist(Polygon_get_point(self,aa),Polygon_get_point(self,bb));
            };
            return length;""")


  ctx= xo.ContextCPU()
  poly = Polygon(points=10,edges=10, _context=ctx)
  poly.points.x=np.random.rand(10);
  poly.points.y=np.random.rand(10);
  poly.points.z=np.random.rand(10);
  poly.points.edges=np.c_[np.arange(10),np.roll(np.arange(10),1)]
  print(poly.get_length())


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

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo  # import


class Point(xo.Struct):  # declare Struct
    x = xo.Float64
    y = xo.Float64


ArrNPoint = Point[:]  # declare Array


source = """
#include <math.h> //only_for_context cpu
#define SQ(x) x*x

/*gpukern*/
void length(/*gpugblmem*/ ArrNPoint pp,
                   double* res){
    /*interate over ii*/
    for (int ii=1; ii<ArrNPoint_len1(pp); ii++){
        res=sqrt(SQ(ArrNPoint_get_x(pp,ii-1)-ArrNPoint_get_x(pp,ii))+
                       SQ(ArrNPoint_get_y(pp,ii-1)-ArrNPoint_get_y(pp,ii)));
    /*iterate over ii*/}
"""


points = ArrNPoint(10)  # instantiate Array

for ii in range(len(points)):  # fill Array
    points[ii].x = ii
    points[ii].y = ii

ctx = xo.ContextCpu()

kernels = {
    "length": xo.Kernel([xo.Arg(ArrNPoint, name="obj"), xo.Arg(xo.Float64[:])])
}

ctx.add_kernels(sources=[source], kernels=kernels)

res = np.zeros(len(points))

/* copyright ##################################
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### */

__kernel void copy_array_fcont(
             const int    fcont, // bool not accepted
             const int    ndim,
             const int    nelem,
    __global const int*   shape,
             const int    itemsize,
    __global const char*  buffer_src,
    __global const int*   strides_src,
             const int    offset_src,
    __global       char*  buffer_dest,
    __global const int*   strides_dest,
             const int    offset_dest
) {
    int gid = get_global_id(0);
    int ibyte, idim, this_shape, slice_size, this_index, flat_index;
    int pos_src, pos_dest, this_stride_src, this_stride_dest;

    slice_size = nelem;
    flat_index = gid;
    pos_src = offset_src;
    pos_dest = offset_dest;
    for (idim=0; idim<ndim; idim++) {
        if (fcont) {
            this_shape = shape[ndim-idim-1];              // for f contiguous
            this_stride_src = strides_src[ndim-idim-1];   // for f contiguous
            this_stride_dest = strides_dest[ndim-idim-1]; // for f contiguous
        }
        else {
            this_shape = shape[idim];              // for c contiguous
            this_stride_src = strides_src[idim];   // for c contiguous
            this_stride_dest = strides_dest[idim]; // for c contiguous
        }

        slice_size = slice_size/this_shape;
        this_index = flat_index/slice_size;
        flat_index = flat_index - this_index*slice_size;

        pos_src = pos_src + this_index * this_stride_src;
        pos_dest = pos_dest + this_index * this_stride_dest;
    }

    for (ibyte=0; ibyte<itemsize; ibyte++) {
        buffer_dest[pos_dest + ibyte] = buffer_src[pos_src + ibyte];
    }
}
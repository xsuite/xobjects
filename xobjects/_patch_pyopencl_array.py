import numpy as np


def _patch_pyopencl_array(cl, cla, ctx):
    prg = cl.Program(
        ctx,
        """
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
                          )
        {
          int gid = get_global_id(0);
          int ibyte, idim, this_shape, slice_size, this_index, flat_index;
          int pos_src, pos_dest, this_stride_src, this_stride_dest;

          slice_size = nelem;
          flat_index = gid;
          pos_src = offset_src;
          pos_dest = offset_dest;
          for (idim=0; idim<ndim; idim++){
            if (fcont){
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

          for (ibyte=0; ibyte<itemsize; ibyte++){
            buffer_dest[pos_dest + ibyte] = buffer_src[pos_src + ibyte];
            }
        }
        """,
    ).build()

    knl_copy_array_fcont = prg.copy_array_fcont

    def _infer_fccont(arr):
        if arr.strides[0] < arr.strides[-1]:
            return "F"
        else:
            return "C"

    def copy_non_cont(src, dest, custom_itemsize=None, skip_typecheck=False):

        assert src.shape == dest.shape

        # The case float -> complex just works (by using the src itemsize)
        if not (src.dtype == np.float64 and dest.dtype == np.complex128):
            if not skip_typecheck:
                assert src.dtype == dest.dtype

        if src.strides[0] != src.strides[-1]:  # check is needed for 1d arrays
            assert _infer_fccont(src) == _infer_fccont(dest)

        if custom_itemsize is not None:
            itemsize = np.int32(custom_itemsize)
        else:
            itemsize = np.int32(src.dtype.itemsize)

        fcontiguous = 0
        if _infer_fccont(dest) == "F":
            fcontiguous = 1
        fcont = np.int32(fcontiguous)
        shape = cla.to_device(dest.queue, np.array(src.shape, dtype=np.int32))
        ndim = np.int32(len(shape))
        nelem = np.int32(np.prod(src.shape))
        buffer_src = src.base_data
        strides_src = cla.to_device(
            dest.queue, np.array(src.strides, dtype=np.int32)
        )
        offset_src = np.int32(src.offset)
        buffer_dest = dest.base_data
        strides_dest = cla.to_device(
            dest.queue, np.array(dest.strides, dtype=np.int32)
        )
        offset_dest = np.int32(dest.offset)

        event = knl_copy_array_fcont(
            dest.queue,
            (nelem,),
            None,
            # args:
            fcont,
            ndim,
            nelem,
            shape.data,
            itemsize,
            buffer_src,
            strides_src.data,
            offset_src,
            buffer_dest,
            strides_dest.data,
            offset_dest,
        )
        event.wait()

    def mysetitem(self, *args, **kwargs):
        try:
            self._old_setitem(*args, **kwargs)
        except (NotImplementedError, ValueError):
            dest = self[args[0]]
            src = args[1]
            if np.isscalar(src):
                src = dest._cont_zeros_like_me() + src
            copy_non_cont(src, dest)

    def mycopy(self):
        res = self._cont_zeros_like_me()
        copy_non_cont(self, res)
        return res

    def myreal(self):
        assert self.dtype == np.complex128
        res = cla.zeros(
            self.queue,
            shape=self.shape,
            dtype=np.float64,
            order=_infer_fccont(self),
        )
        copy_non_cont(self, res, custom_itemsize=8, skip_typecheck=True)
        return res

    def myget(self):
        try:
            return self._old_get()
        except AssertionError:
            return self.copy().get()

    def _cont_zeros_like_me(self):
        res = cla.zeros(
            self.queue,
            shape=self.shape,
            dtype=self.dtype,
            order=_infer_fccont(self),
        )
        return res

    # sum not implemented by pyopencl, I add it
    def mysum(self):
        dtype = getattr(np, self.dtype.name)
        try:
            res = dtype(cla.sum(self).get())
        except RuntimeError:
            res = dtype(cla.sum(self.copy()).get())

        return res

    cla.Array._cont_zeros_like_me = _cont_zeros_like_me

    if not hasattr(cla.Array, "_old_copy"):
        cla.Array._old_copy = cla.Array.copy
    cla.Array.copy = mycopy

    if not hasattr(cla.Array, "_old_setitem"):
        cla.Array._old_setitem = cla.Array.__setitem__
    cla.Array.__setitem__ = mysetitem

    if not hasattr(cla.Array, "_old_get"):
        cla.Array._old_get = cla.Array.get
    cla.Array.get = myget

    cla.Array.real = property(myreal)
    cla.Array.sum = mysum

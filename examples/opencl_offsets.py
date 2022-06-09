# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import pyopencl as cl
import pyopencl.array as cl_array


def try_offset(ctx):
    queue = cl.CommandQueue(ctx)

    a = cl_array.Array(queue, shape=(2048,), dtype="uint8")
    b = a[3:]

    try:
        b.data
    except Exception as ex:
        pass
        # print("Always fails due to ArrayHasOffsetError")
        # print(repr(ex))

    for offset in 3, 4, 8, 16, 32, 64, 128:
        try:
            b = a[offset:]
            b.base_data[b.offset : b.offset + b.nbytes]
            print(f"offset={b.offset} OK")
        except Exception as ex:
            # Fails in some platform due to clCreateSubBuffer MISALIGNED_SUB_BUFFER_OFFSET"
            print(f"offset={b.offset} not OK")
            # print(repr(ex))


for platform in cl.get_platforms():
    for device in platform.get_devices():
        ctx = cl.Context(devices=[device])
        print(f"{ctx.devices[0].name} on {ctx.devices[0].platform.name}")
        try_offset(ctx)

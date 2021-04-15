import pyopencl as cl
import pyopencl.array as cl_array


def try_offset(ctx):
    queue = cl.CommandQueue(ctx)

    a = cl_array.Array(queue, shape=(10,), dtype="uint8")
    b = a[3:]

    try:
        b.data
    except Exception as ex:
        print("Always fails due to ArrayHasOffsetError")
        print(repr(ex))

    try:
        b = a[3:]
        b.base_data[b.offset : b.offset + b.nbytes]
        print(f"offset={b.offset} OK")
    except Exception as ex:
        print(f"offset={b.offset} not OK")
        print(
            "Fails in some platform due to clCreateSubBuffer MISALIGNED_SUB_BUFFER_OFFSET"
        )
        print(repr(ex))

    try:
        b = a[4:]
        b.base_data[b.offset : b.offset + b.nbytes]
        print(f"offset={b.offset} OK")
    except Exception as ex:
        print(f"offset={b.offset} not OK")
        print("Does not fail easily, but still can fail")
        print(repr(ex))

    try:
        b = a[8:]
        b.base_data[b.offset : b.offset + b.nbytes]
        print(f"offset={b.offset} OK")
    except Exception as ex:
        print(f"offset={b.offset} not OK")
        print("Does not fail easily, but still can fail")
        print(repr(ex))


for platform in cl.get_platforms():
    for device in platform.get_devices():
        ctx = cl.Context(devices=[device])
        print(ctx)
        try_offset(ctx)

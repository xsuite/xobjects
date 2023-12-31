import xobjects as xo


import pytest

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_opencl_device(test_context):
    if not isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Pyopencl not yet supported for footprint")
        return

    xo.ContextPyopencl.print_devices()
    for device in xo.ContextPyopencl.get_devices():
        ctx = xo.ContextPyopencl(device=device)
        ctx = xo.ContextPyopencl(device=ctx.device)

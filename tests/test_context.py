# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2022.                   #
# ########################################### #
import shutil
import sys

import cffi
import pytest

import xobjects as xo


@pytest.fixture
def cleanup_test_module():
    yield
    shutil.rmtree('./some_test')


def test_context_cpu_add_kernels_retain(mocker, cleanup_test_module):
    context = xo.ContextCpu()

    test_src = """
        int32_t test_function() {
            return 42;
        }
    """

    kernels = {
        'test_function': xo.Kernel(
            args=[],
            c_name='test_function',
            ret=xo.Arg(xo.Int32),
        )
    }

    context.add_kernels(
        sources=[test_src],
        kernels=kernels,
        built_ffi_module_name='some_test.package.test_module',
        compile=True,
    )

    # Check if the kernel works
    assert context.kernels.test_function() == 42

    cffi_compile = mocker.patch.object(cffi.FFI, 'compile')
    fresh_context = xo.ContextCpu()
    fresh_context.add_kernels(
        kernels=kernels,
        built_ffi_module_name='some_test.package.test_module',
        compile=False,
    )

    # Check that the new kernel works
    assert fresh_context.kernels.test_function() == 42

    # And that it was not recompiled
    cffi_compile.assert_not_called()


@pytest.fixture
def fake_existing_package(tmp_path, mocker):
    original_path = sys.path.copy()
    sys.path.append(str(tmp_path))

    sys.path.append(str(tmp_path))
    test_package = tmp_path / 'test_package'
    test_package.mkdir()
    init = test_package / '__init__.py'
    init.write_text('')

    yield test_package

    sys.path = original_path


def test_context_cpu_add_kernels_existing_package(fake_existing_package):
    context = xo.ContextCpu()

    test_src = """
            int32_t test_function() {
                return 7;
            }
        """

    kernels = {
        'test_function': xo.Kernel(
            args=[],
            c_name='test_function',
            ret=xo.Arg(xo.Int32),
        )
    }

    context.add_kernels(
        sources=[test_src],
        kernels=kernels,
        built_ffi_module_name='test_package.test_module',
        compile=True,
    )

    # Check that the generated module can be used
    import test_package.test_module  # noqa

    # Check if the kernel works
    assert context.kernels.test_function() == 7

    # Check that the dll was created in the right place
    dll_exists = False
    for file in fake_existing_package.iterdir():
        if not file.name.startswith('test_module'):
            continue
        if file.suffix not in ['.so', '.dll', '.dylib', '.pyd']:
            continue
        dll_exists = True
    assert dll_exists

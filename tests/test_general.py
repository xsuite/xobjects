import os
import subprocess
import sys

import pytest

import xobjects as xo
from xobjects.general import Print
from xobjects.test_helpers import allow_kernel_compilation


def test_print_mode_default(capsys):
    printer = Print()

    printer("visible")

    assert capsys.readouterr().out == "visible\n"


def test_print_mode_suppressed(capsys):
    printer = Print()
    with xo.settings.override(print_mode="suppress"):
        printer("hidden")

    assert capsys.readouterr().out == ""


def test_settings_control_xsuite_printer(capsys):
    with xo.settings.override(print_mode="suppress"):
        xo._print("hidden")

    assert capsys.readouterr().out == ""


def test_python_setting_overrides_environment_default(capsys):
    with xo.settings.override(print_mode="print"):
        xo._print("visible")

    assert capsys.readouterr().out == "visible\n"


def test_invalid_print_mode_setting():
    with pytest.raises(ValueError, match="XSUITE_PRINT_MODE.*print.*suppress"):
        xo.settings.print_mode = "invalid"


def test_settings_override_restores_value_after_error():
    original = xo.settings.print_mode

    with pytest.raises(RuntimeError):
        with xo.settings.override(print_mode="suppress"):
            assert xo.settings.print_mode == "suppress"
            raise RuntimeError

    assert xo.settings.print_mode == original


def test_settings_are_discoverable():
    expected_settings = {
        "print_mode",
        "progress_indicator",
        "allow_kernel_compilation",
        "force_kernel_compilation",
        "show_kernel_diagnostics",
        "cffi_forbid_compile",
        "cffi_keep_build_files",
        "cuda_backend",
        "cuda_fast_compile",
        "cuda_compiler",
    }

    assert expected_settings <= set(dir(xo.settings))
    for name in expected_settings:
        assert f"{name}=" in repr(xo.settings)
        assert name in type(xo.settings).__doc__


def test_print_mode_environment_variable_is_startup_default():
    environment = os.environ.copy()
    environment["XSUITE_PRINT_MODE"] = "suppress"
    code = (
        "import xobjects as xo; "
        'assert xo.settings.print_mode == "suppress"; '
        'xo._print("hidden")'
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout == ""


def test_print_mode_environment_variable_can_be_overridden_in_python():
    environment = os.environ.copy()
    environment["XSUITE_PRINT_MODE"] = "suppress"
    code = (
        "import xobjects as xo; "
        'xo.settings.print_mode = "print"; '
        'assert xo.settings.print_mode == "print"; '
        'xo._print("visible")'
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout == "visible\n"


def test_environment_change_after_import_does_not_override_python_setting(
    capsys, monkeypatch
):
    with xo.settings.override(print_mode="print"):
        monkeypatch.setenv("XSUITE_PRINT_MODE", "suppress")
        xo._print("visible")

    assert capsys.readouterr().out == "visible\n"


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_boolean_environment_true_values(value):
    environment = os.environ.copy()
    environment["XSUITE_ALLOW_KERNEL_COMPILATION"] = value
    code = (
        "import xobjects as xo; "
        "assert xo.settings.allow_kernel_compilation is True; "
        "xo.settings.allow_kernel_compilation = False; "
        "assert xo.settings.allow_kernel_compilation is False"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_boolean_environment_false_values(value):
    environment = os.environ.copy()
    environment["XSUITE_FORCE_KERNEL_COMPILATION"] = value
    code = (
        "import xobjects as xo; "
        "assert xo.settings.force_kernel_compilation is False"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_invalid_boolean_environment_value():
    environment = os.environ.copy()
    environment["XSUITE_CFFI_FORBID_COMPILE"] = "sometimes"

    completed = subprocess.run(
        [sys.executable, "-c", "import xobjects"],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Invalid boolean value" in completed.stderr
    assert "xobjects.settings.cffi_forbid_compile" in completed.stderr
    assert "XSUITE_CFFI_FORBID_COMPILE" in completed.stderr


def test_runtime_settings_environment_defaults():
    environment = os.environ.copy()
    environment.update(
        {
            "XSUITE_PROGRESS_INDICATOR": "text",
            "XSUITE_FORCE_KERNEL_COMPILATION": "yes",
            "XSUITE_SHOW_KERNEL_DIAGNOSTICS": "on",
            "XSUITE_CFFI_FORBID_COMPILE": "true",
            "XSUITE_CFFI_KEEP_BUILD_FILES": "1",
            "XSUITE_CUDA_BACKEND": "clang",
            "XSUITE_CUDA_FAST_COMPILE": "false",
            "XSUITE_CUDA_COMPILER": "/path/to/clang++",
        }
    )
    code = (
        "import xobjects as xo; "
        'assert xo.settings.progress_indicator == "text"; '
        "assert xo.settings.force_kernel_compilation is True; "
        "assert xo.settings.show_kernel_diagnostics is True; "
        "assert xo.settings.cffi_forbid_compile is True; "
        "assert xo.settings.cffi_keep_build_files is True; "
        'assert xo.settings.cuda_backend == "clang"; '
        "assert xo.settings.cuda_fast_compile is False; "
        'assert xo.settings.cuda_compiler == "/path/to/clang++"'
    )

    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize(
    "allow, force, compilation_allowed",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_kernel_compilation_settings(allow, force, compilation_allowed):
    with xo.settings.override(
        allow_kernel_compilation=allow,
        force_kernel_compilation=force,
    ):
        assert (
            xo.context_cpu.kernel_compilation_allowed(xo.ContextCpu())
            is compilation_allowed
        )


def test_user_context_environment_variable(monkeypatch):
    monkeypatch.setenv("XOBJECTS_USER_CONTEXT", "ContextCpu:auto")
    context = xo.get_user_context()

    assert context.openmp_enabled


def test_test_contexts_environment_variable(monkeypatch):
    monkeypatch.setenv(
        "XOBJECTS_TEST_CONTEXTS",
        "ContextCpu;ContextCpu:auto",
    )

    contexts = list(xo.context.get_test_contexts())

    assert len(contexts) == 2
    assert contexts[0].openmp_enabled is False
    assert contexts[1].openmp_enabled is True


def test_cffi_forbid_compile_setting():
    with xo.settings.override(cffi_forbid_compile=True):
        with pytest.raises(RuntimeError) as err:
            xo.ContextCpu().build_kernels({})

    message = str(err.value)
    assert "xobjects.settings.cffi_forbid_compile" in message
    assert "XSUITE_CFFI_FORBID_COMPILE" in message


def test_allow_kernel_compilation_decorator_restores_state():
    @allow_kernel_compilation(skip_when_forbid_compile=False)
    def decorated():
        assert xo.settings.allow_kernel_compilation is True

    with xo.settings.override(allow_kernel_compilation=False):
        decorated()
        assert xo.settings.allow_kernel_compilation is False

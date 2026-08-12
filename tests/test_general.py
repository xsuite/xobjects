import os
import subprocess
import sys

import pytest

import xobjects as xo
from xobjects.general import Print
from xobjects.test_helpers import allow_no_prebuilt_kernels


def test_print_mode_default(capsys):
    printer = Print()

    printer('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_print_mode_suppressed(capsys):
    printer = Print()
    with xo.settings.override(print_mode='suppress'):
        printer('hidden')

    assert capsys.readouterr().out == ''


def test_settings_control_xsuite_printer(capsys):
    with xo.settings.override(print_mode='suppress'):
        xo._print('hidden')

    assert capsys.readouterr().out == ''


def test_python_setting_overrides_environment_default(capsys):
    with xo.settings.override(print_mode='print'):
        xo._print('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_legacy_suppress_overrides_mode(capsys):
    printer = Print()
    printer.suppress = True

    printer('hidden')

    assert capsys.readouterr().out == ''


def test_invalid_print_mode_setting():
    with pytest.raises(ValueError, match='expected.*print.*suppress'):
        xo.settings.print_mode = 'invalid'


def test_settings_override_restores_value_after_error():
    original = xo.settings.print_mode

    with pytest.raises(RuntimeError):
        with xo.settings.override(print_mode='suppress'):
            assert xo.settings.print_mode == 'suppress'
            raise RuntimeError

    assert xo.settings.print_mode == original


def test_settings_are_discoverable():
    assert 'print_mode' in dir(xo.settings)
    assert 'allow_no_prebuilt_kernels' in dir(xo.settings)
    assert 'print_mode=' in repr(xo.settings)


def test_print_mode_environment_variable_is_startup_default():
    environment = os.environ.copy()
    environment['XSUITE_PRINT_MODE'] = 'suppress'
    code = (
        'import xobjects as xo; '
        'assert xo.settings.print_mode == "suppress"; '
        'xo._print("hidden")')

    completed = subprocess.run(
        [sys.executable, '-c', code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout == ''


def test_print_mode_environment_variable_can_be_overridden_in_python():
    environment = os.environ.copy()
    environment['XSUITE_PRINT_MODE'] = 'suppress'
    code = (
        'import xobjects as xo; '
        'xo.settings.print_mode = "print"; '
        'assert xo.settings.print_mode == "print"; '
        'xo._print("visible")')

    completed = subprocess.run(
        [sys.executable, '-c', code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout == 'visible\n'


def test_environment_change_after_import_does_not_override_python_setting(
        capsys, monkeypatch):
    with xo.settings.override(print_mode='print'):
        monkeypatch.setenv('XSUITE_PRINT_MODE', 'suppress')
        xo._print('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_allow_no_prebuilt_kernels_environment_is_startup_default():
    environment = os.environ.copy()
    environment['XSUITE_ALLOW_NO_PREBUILT_KERNELS'] = '1'
    code = (
        'import xobjects as xo; '
        'assert xo.settings.allow_no_prebuilt_kernels is True; '
        'xo.settings.allow_no_prebuilt_kernels = False; '
        'assert xo.settings.allow_no_prebuilt_kernels is False')

    subprocess.run(
        [sys.executable, '-c', code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_allow_no_prebuilt_kernels_decorator_restores_state(monkeypatch):
    monkeypatch.delenv('XSUITE_ALLOW_NO_PREBUILT_KERNELS', raising=False)

    @allow_no_prebuilt_kernels(skip_when_forbid_compile=False)
    def decorated():
        assert xo.settings.allow_no_prebuilt_kernels is True
        assert os.environ['XSUITE_ALLOW_NO_PREBUILT_KERNELS'] == '1'
        subprocess.run(
            [
                sys.executable,
                '-c',
                ('import xobjects as xo; assert '
                 'xo.settings.allow_no_prebuilt_kernels is True'),
            ],
            check=True,
        )

    with xo.settings.override(allow_no_prebuilt_kernels=False):
        decorated()
        assert xo.settings.allow_no_prebuilt_kernels is False
        assert 'XSUITE_ALLOW_NO_PREBUILT_KERNELS' not in os.environ

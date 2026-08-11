import pytest

from xobjects.general import Print


def test_print_mode_default(capsys, monkeypatch):
    monkeypatch.delenv('XSUITE_PRINT_MODE', raising=False)
    printer = Print()

    printer('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_print_mode_suppressed(capsys, monkeypatch):
    monkeypatch.delenv('XSUITE_PRINT_MODE', raising=False)
    printer = Print()
    printer.mode = 'suppress'

    printer('hidden')

    assert capsys.readouterr().out == ''


def test_print_mode_environment_variable_takes_precedence(
        capsys, monkeypatch):
    printer = Print()
    printer.mode = 'suppress'
    monkeypatch.setenv('XSUITE_PRINT_MODE', 'print')

    printer('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_print_mode_environment_variable_suppresses(capsys, monkeypatch):
    printer = Print()
    monkeypatch.setenv('XSUITE_PRINT_MODE', 'suppress')

    printer('hidden')

    assert capsys.readouterr().out == ''


def test_legacy_suppress_overrides_mode(capsys, monkeypatch):
    monkeypatch.setenv('XSUITE_PRINT_MODE', 'print')
    printer = Print()
    printer.suppress = True

    printer('hidden')

    assert capsys.readouterr().out == ''


def test_invalid_print_mode_environment_variable(monkeypatch):
    monkeypatch.setenv('XSUITE_PRINT_MODE', 'invalid')

    with pytest.raises(ValueError, match='expected.*print.*suppress'):
        Print()('invalid')


def test_invalid_print_mode_module_attribute(monkeypatch):
    monkeypatch.delenv('XSUITE_PRINT_MODE', raising=False)
    printer = Print()
    printer.mode = 'invalid'

    with pytest.raises(ValueError, match='expected.*print.*suppress'):
        printer('invalid')

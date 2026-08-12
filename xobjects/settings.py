import os
from contextlib import contextmanager


class Settings:
    """Process-wide settings shared by the Xsuite packages."""

    def __init__(self):
        object.__setattr__(self, '_definitions', {})
        object.__setattr__(self, '_values', {})

    def _register(
        self,
        name,
        *,
        default,
        environment_variable=None,
        choices=None,
        environment_parser=None,
        on_change=None,
        getter=None,
    ):
        if name in self._definitions:
            raise ValueError(f'Setting {name!r} is already registered.')

        definition = {
            'environment_variable': environment_variable,
            'choices': choices,
            'on_change': on_change,
            'getter': getter,
        }
        self._definitions[name] = definition

        value = default
        if (environment_variable is not None
                and environment_variable in os.environ):
            environment_value = os.environ[environment_variable]
            value = (environment_parser(environment_value)
                     if environment_parser else environment_value)
        self._set(name, value)

    def _set(self, name, value):
        try:
            definition = self._definitions[name]
        except KeyError as err:
            raise AttributeError(f'Unknown Xsuite setting {name!r}.') from err

        choices = definition['choices']
        if choices is not None and value not in choices:
            expected = ', '.join(repr(choice) for choice in choices)
            raise ValueError(
                f'Invalid value {value!r} for setting {name!r}; '
                f'expected one of {expected}.')

        self._values[name] = value
        on_change = definition['on_change']
        if on_change is not None:
            on_change(value)

    def __getattr__(self, name):
        try:
            definition = self._definitions[name]
        except KeyError as err:
            raise AttributeError(f'Unknown Xsuite setting {name!r}.') from err
        getter = definition['getter']
        return getter() if getter is not None else self._values[name]

    def __setattr__(self, name, value):
        self._set(name, value)

    @contextmanager
    def override(self, **kwargs):
        """Temporarily override settings and restore them on exit."""
        previous = {}
        for name, value in kwargs.items():
            if name not in self._definitions:
                raise AttributeError(f'Unknown Xsuite setting {name!r}.')
            previous[name] = getattr(self, name)

        # Validate every value before changing any setting.
        for name, value in kwargs.items():
            choices = self._definitions[name]['choices']
            if choices is not None and value not in choices:
                expected = ', '.join(repr(choice) for choice in choices)
                raise ValueError(
                    f'Invalid value {value!r} for setting {name!r}; '
                    f'expected one of {expected}.')

        try:
            for name, value in kwargs.items():
                self._set(name, value)
            yield self
        finally:
            for name, value in previous.items():
                self._set(name, value)

    def __repr__(self):
        values = ', '.join(
            f'{name}={getattr(self, name)!r}' for name in self._values)
        return f'Settings({values})'

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._definitions))


settings = Settings()
settings._register(
    'print_mode',
    default='print',
    environment_variable='XSUITE_PRINT_MODE',
    choices=('print', 'suppress'),
)
settings._register(
    'allow_no_prebuilt_kernels',
    default=False,
    environment_variable='XSUITE_ALLOW_NO_PREBUILT_KERNELS',
    choices=(False, True),
    environment_parser=lambda value: True,
)

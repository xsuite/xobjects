# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2022.                   #
# ########################################### #

from functools import wraps
from typing import Callable, Iterable, Union

import pytest

from .context import get_context_from_string, get_test_contexts


def _for_all_test_contexts_excluding(
    test_function: Callable,
    excluding: Union[Iterable[str], str] = (),
) -> Callable:
    """Parametrize the decorated test over all test contexts with the argument
    `test_context`, excluding those contexts whose names are in ``excluding."""
    if isinstance(excluding, str):
        excluding = (excluding,)

    test_context_names = tuple(
        ctx_name
        for ctx in get_test_contexts()
        if (ctx_name := type(ctx).__name__) not in excluding
    )

    @wraps(test_function)
    def actual_test(*args, **kwargs):
        kwargs["test_context"] = get_context_from_string(
            kwargs["test_context"]
        )
        test_function(*args, **kwargs)

    if len(test_context_names) == 0:
        return pytest.mark.skip(
            "All available contexts have been excluded for "
            f"this test: {excluding}."
        )(actual_test)

    return pytest.mark.parametrize(
        "test_context",
        test_context_names,
    )(actual_test)


def for_all_test_contexts(*args, **kwargs):
    """Parametrize the decorated test over all test contexts with the argument
    `test_context`, excluding those contexts whose names are in `excluding`.

    Can be used in both of the below forms:

    @for_all_test_contexts
    def test_all(test_context):
        ...

    @for_all_test_contexts(excluding=('ContextPyopencl',))
    def test_all_but_opencl(test_context):
        ...
    """
    if len(args) == 1 and not kwargs and callable(args[0]):
        return _for_all_test_contexts_excluding(args[0])
    elif not args and len(kwargs) == 1 and "excluding" in kwargs:

        def decorator(test_function):
            return _for_all_test_contexts_excluding(
                test_function, excluding=kwargs["excluding"]
            )

        return decorator

    raise ValueError(
        f"@for_all_test_contexts can only be used either directly "
        f"on the test, or with a single argument `excluding`."
    )


def requires_context(context_name: str):
    ctx_names = (type(ctx).__name__ for ctx in get_test_contexts())

    if {context_name} & set(ctx_names):  # proceed as normal
        return lambda test_function: test_function

    return pytest.mark.skip(f"{context_name} is unavailable on this platform.")

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2022.                   #
# ########################################### #

from contextlib import contextmanager
from functools import wraps
from typing import List

import pytest

from .context import get_context_from_string, get_test_contexts


def for_all_test_contexts(test_function):
    """Parametrize the decorated test over all test contexts with the argument
    `test_context`."""
    test_context_names = (type(ctx).__name__ for ctx in get_test_contexts())

    @wraps(test_function)
    def actual_test(*args, **kwargs):
        kwargs['test_context'] = get_context_from_string(kwargs['test_context'])
        test_function(*args, **kwargs)

    return pytest.mark.parametrize(
        'test_context',
        test_context_names,
    )(actual_test)


def requires_context(context_name: str):
    ctx_names = (type(ctx).__name__ for ctx in get_test_contexts())

    if {context_name} & set(ctx_names):  # proceed as normal
        return lambda test_function: test_function

    return pytest.mark.skip(f'Unavailable on this platform: {context_name}')

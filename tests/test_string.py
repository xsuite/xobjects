# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


def test_string_class():
    info = xo.String._inspect_args("abcde")
    assert info.size == 5 + 1 + 2 + 8


def test_string_fixed():
    cls = xo.String.fixed(10)
    assert cls.__name__ == "String10"
    assert cls._size == 10


def test_string_init1():
    ss = xo.String(10)
    assert ss._size is not None
    assert ss._buffer.capacity == 18


def test_string_init2():
    ss = xo.String("test")
    assert ss._size is not None
    assert ss.to_str() == "test"


@for_all_test_contexts
def test_string_init3(test_context):
    ss = xo.String("test", _context=test_context)
    assert xo.String._from_buffer(ss._buffer, ss._offset) == "test"


@for_all_test_contexts
def test_string_array(test_context):
    import numpy as np

    StringArray = xo.String[:]
    pdata = ["asd", "as"]
    npdata = np.array(pdata)

    xobj = StringArray(npdata, _context=test_context)
    assert npdata[1] == xobj[1]

    xobj = StringArray(pdata, _context=test_context)
    assert pdata[1] == xobj[1]

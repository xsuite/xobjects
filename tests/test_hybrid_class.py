# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import pytest

import numpy as np
import xobjects as xo

def test_hybrid_struct():

    class Element(xo.HybridClass):
        _xofields = {
            'n': xo.Int32,
            'b': xo.Float64,
            'vv': xo.Float64[:],
        }
        def __init__(self, vv=None, **kwargs):
            if "_xobject" in kwargs.keys():
                self.xoinitialize(**kwargs)
            else:
                self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv, **kwargs)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        ele = Element([1, 2, 3], _context=context)
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 2

        new_vv = context.nparray_to_context_array(np.array([7, 8, 9]))
        ele.vv = new_vv
        assert ele.n == ele._xobject.n == 3
        assert ele.b == ele._xobject.b == 6
        assert ele.vv[1] == ele._xobject.vv[1] == 8

        ele.n = 5.0
        assert ele.n == ele._xobject.n == 5

        ele.b = 50
        assert ele.b == ele._xobject.b == 50.0

        dd = ele.to_dict()
        assert dd["vv"][1] == 8
        assert isinstance(dd["vv"], np.ndarray)


def test_explicit_buffer():

    class Element(xo.HybridClass):
        _xofields = {
            'n': xo.Int32,
            'b': xo.Float64,
            'vv': xo.Float64[:],
        }
        def __init__(self, vv=None, **kwargs):
            self.xoinitialize(n=len(vv), b=np.sum(vv), vv=vv, **kwargs)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        ele1 = Element([1, 2, 3], _context=context)
        ele2 = Element([7, 8, 9], _buffer=ele1._buffer)

        assert ele1.vv[1] == ele1._xobject.vv[1] == 2
        assert ele2.vv[1] == ele2._xobject.vv[1] == 8
        for ee in [ele1, ele2]:
            assert ee._buffer is ee._xobject._buffer
            assert ee._offset == ee._xobject._offset

        assert ele1._buffer is ele2._buffer
        assert ele1._offset != ele2._offset

@pytest.fixture
def classes_for_test_hybrid_class_no_ref():
    class InnerClass(xo.HybridClass):
        _xofields = {
            'a': xo.Int64,
            'b': xo.Float64[:],
        }

    class OuterClass(xo.HybridClass):
        _xofields = {
            'inner': InnerClass,
            'inner_to_rename': InnerClass,
            's': xo.Float64,
        }

        _rename = {'inner_to_rename': 'inner_renamed'}

    return InnerClass, OuterClass

def test_nested_hybrid_init_no_ref(classes_for_test_hybrid_class_no_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_no_ref

    inner = InnerClass(a=1, b=[2, 3, 4])
    inner.z = 45
    initial_xobject = inner._xobject
    outer = OuterClass(inner=inner, inner_renamed=inner)

    assert inner._xobject is initial_xobject
    assert outer.inner.z == inner.z
    assert outer._buffer is outer.inner._buffer

    inner.b[1] = 1000
    assert inner.b[1] == 1000
    assert outer.inner.b[1] == 3

    outer.inner.z = 200
    assert outer.inner.z == 200
    assert inner.z == 45


def test_nested_hybrid_setattr_no_ref(classes_for_test_hybrid_class_no_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_no_ref

    inner = InnerClass(a=1, b=[2, 3, 4])
    inner.z = 45
    initial_xobject = inner._xobject
    outer = OuterClass(inner={'b': 3}, inner_renamed={'b': 3})

    outer.inner = inner
    outer.inner_renamed = inner

    assert inner._xobject is initial_xobject
    assert outer.inner.z == inner.z
    assert outer._buffer is outer.inner._buffer

    inner.b[1] = 1000
    assert inner.b[1] == 1000
    assert outer.inner.b[1] == 3

    outer.inner.z = 200
    assert outer.inner.z == 200
    assert inner.z == 45


@pytest.fixture
def classes_for_test_hybrid_class_ref():
    class InnerClass(xo.HybridClass):
        _xofields = {
            'a': xo.Int64,
            'b': xo.Float64[:],
        }

    class OuterClass(xo.HybridClass):
        _xofields = {
            'inner': xo.Ref(InnerClass),
            'inner_to_rename': xo.Ref(InnerClass),
            's': xo.Float64,
        }

        _rename = {'inner_to_rename': 'inner_renamed'}

    return InnerClass, OuterClass


def test_nested_hybrid_init_with_ref(classes_for_test_hybrid_class_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_ref

    buf = xo.context_default.new_buffer()

    inner = InnerClass(a=1, b=[2, 3, 4], _buffer=buf)
    inner.z = 45
    outer1 = OuterClass(inner=inner, inner_renamed=inner, _buffer=buf)
    outer2 = OuterClass(inner=inner, inner_renamed=inner, _buffer=buf)

    assert inner._buffer is buf
    assert outer1.inner._buffer is buf
    assert outer2.inner._buffer is buf
    assert outer1.inner._offset == inner._offset
    assert outer2.inner._offset == inner._offset

    assert outer1.inner_renamed._buffer is buf
    assert outer2.inner_renamed._buffer is buf
    assert outer1.inner_renamed._offset == inner._offset
    assert outer2.inner_renamed._offset == inner._offset

    assert outer1.inner is outer1._dressed_inner
    assert outer1.inner_renamed is outer1._dressed_inner_to_rename
    assert outer2.inner is outer2._dressed_inner
    assert outer2.inner_renamed is outer2._dressed_inner_to_rename

    outer1.inner.z = 100
    assert outer1.inner.z == 100
    assert outer2.inner.z == 100


def test_nested_hybrid_init_with_ref_different_buf(classes_for_test_hybrid_class_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_ref

    buf_inner = xo.context_default.new_buffer()
    buf_outer = xo.context_default.new_buffer()

    inner = InnerClass(a=1, b=[2, 3, 4], _buffer=buf_inner)

    with pytest.raises(MemoryError):
        outer = OuterClass(inner=inner, inner_renamed=inner, _buffer=buf_outer)


def test_nested_hybrid_setattr_with_ref(classes_for_test_hybrid_class_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_ref

    buf = xo.context_default.new_buffer()

    inner = InnerClass(a=1, b=[2, 3, 4], _buffer=buf)
    inner.z = 45
    outer1 = OuterClass(inner={'b': 3}, inner_renamed={'b': 3}, _buffer=buf)
    outer2 = OuterClass(inner={'b': 3}, inner_renamed={'b': 3}, _buffer=buf)

    outer1.inner = inner
    outer2.inner = inner
    outer1.inner_renamed = inner
    outer2.inner_renamed = inner

    assert inner._buffer is buf
    assert outer1.inner._buffer is buf
    assert outer2.inner._buffer is buf
    assert outer1.inner._offset == inner._offset
    assert outer2.inner._offset == inner._offset

    assert outer1.inner_renamed._buffer is buf
    assert outer2.inner_renamed._buffer is buf
    assert outer1.inner_renamed._offset == inner._offset
    assert outer2.inner_renamed._offset == inner._offset

    assert outer1.inner is outer1._dressed_inner
    assert outer1.inner_renamed is outer1._dressed_inner_to_rename
    assert outer2.inner is outer2._dressed_inner
    assert outer2.inner_renamed is outer2._dressed_inner_to_rename

    outer1.inner.z = 100
    assert outer1.inner.z == 100
    assert outer2.inner.z == 100


def test_rename_of_two_xo_fields_to_same_name_fails():
    with pytest.raises(ValueError):
        class TestClass(xo.HybridClass):
            _xofields = {
                'a': xo.Int64,
                'b': xo.Int64,
            }
            _rename = {
                'a': 'c',
                'b': 'c',
            }


def test_rename_with_ambiguous_fields_fails():
    with pytest.raises(ValueError):
        class TestClass(xo.HybridClass):
            _xofields = {
                'a': xo.Int64,
                'b': xo.Int64,
            }
            _rename = {
                'a': 'b',
                'b': 'c',
            }

def test_move_nested_objects_between_contexts_no_ref(classes_for_test_hybrid_class_no_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_no_ref

    buffer1 = xo.context_default.new_buffer()
    buffer2 = xo.context_default.new_buffer()

    inner = InnerClass(a=2, b=range(10), _buffer=buffer1)
    outer = OuterClass(inner=inner, inner_renamed=inner, _buffer=buffer1)

    outer.move(_buffer=buffer2)

    assert outer._buffer is buffer2
    assert outer.inner._buffer is buffer2

    outer.move(_buffer=buffer1)

    assert outer._buffer is buffer1
    assert outer.inner._buffer is buffer1


def test_move_nested_objects_with_ref_fails(classes_for_test_hybrid_class_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_ref

    inner = InnerClass(a=2, b=range(10))
    outer = OuterClass(inner=inner, inner_renamed=inner, _buffer=inner._buffer)

    different_buffer = xo.context_default.new_buffer()

    with pytest.raises(MemoryError):
        outer.move(_buffer=different_buffer)


def test_move_field_of_nested_fails(classes_for_test_hybrid_class_ref):
    InnerClass, OuterClass = classes_for_test_hybrid_class_ref

    inner = InnerClass(a=2, b=range(10))
    outer = OuterClass(inner=inner, inner_renamed=inner, _buffer=inner._buffer)

    different_buffer = xo.context_default.new_buffer()

    with pytest.raises(MemoryError):
        outer.inner.move(_buffer=different_buffer)

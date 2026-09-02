# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
import pytest

import xobjects as xo


class RawValue(xo.RawUnion):
    scalar = xo.Float64
    bits = xo.UInt64


class Holder(xo.Struct):
    value = RawValue


def _float_bits(value):
    return np.array([value], dtype=np.float64).view(np.uint64)[0]


def test_raw_union_declaration_consumes_member_attributes():
    assert RawValue._members == {"scalar": xo.Float64, "bits": xo.UInt64}
    assert RawValue._size == 8

    assert not hasattr(RawValue, "scalar")
    assert not hasattr(RawValue, "bits")


def test_raw_union_standalone_instance_reads_explicit_members():
    value = 3.5
    raw_value = RawValue(value)

    assert isinstance(raw_value, RawValue)
    assert raw_value.scalar == value
    assert raw_value.bits == _float_bits(value)
    assert "scalar" in dir(raw_value)
    assert "bits" in dir(raw_value)

    with pytest.raises(AttributeError, match="missing"):
        raw_value.missing


def test_raw_union_struct_field_reads_explicit_members():
    value = 3.5
    holder = Holder(value=value)

    assert isinstance(holder.value, RawValue)
    assert holder.value.scalar == value
    assert holder.value.bits == _float_bits(value)

    with pytest.raises(TypeError):
        float(holder.value)


def test_raw_union_constructor_accepts_explicit_member():
    value = 3.5
    value_bits = _float_bits(value)
    holder = Holder(value=("bits", value_bits))

    assert holder.value.scalar == value
    assert holder.value.bits == value_bits


def test_raw_union_copy_from_same_type():
    source = RawValue(3.5)
    holder = Holder(value=0.0)

    holder.value = source
    assert holder.value.scalar == source.scalar
    assert holder.value.bits == source.bits

    copied = RawValue(source)
    assert copied.scalar == source.scalar
    assert copied.bits == source.bits

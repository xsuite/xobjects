# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from hashlib import new
import json
from inspect import isclass

import numpy as np
from .struct import Struct
from .typeutils import context_default

class _FieldOfDressed:
    def __init__(self, name, _XoStruct):
        self.name = name
        self.isnplikearray = False

        fnames = [ff.name for ff in _XoStruct._fields]
        if self.name in fnames:
            ftype = getattr(_XoStruct, self.name).ftype
            if hasattr(ftype, "_itemtype"):  # is xo object
                if hasattr(ftype._itemtype, "_dtype"):  # valid nplike object
                    self.isnplikearray = True

    def __get__(self, container, ContainerType=None):
        if self.isnplikearray:
            if hasattr(container, "_lim_arrays_name"):
                lim = getattr(container, container._lim_arrays_name)
                return getattr(container._xobject, self.name).to_nplike()[:lim]
            else:
                return getattr(container._xobject, self.name).to_nplike()
        elif hasattr(container, "_dressed_" + self.name):
            return getattr(container, "_dressed_" + self.name)
        else:
            return getattr(container._xobject, self.name)

    def __set__(self, container, value):
        if self.isnplikearray:
            self.__get__(container=container)[:] = value
        elif hasattr(value, "_xobject"):  # value is a dressed xobject
            setattr(container, "_dressed_" + self.name, value)
            setattr(container._xobject, self.name, value._xobject)
            getattr(container, self.name)._xobject = getattr(
                container._xobject, self.name
            )
        else:
            self.content = None
            setattr(container._xobject, self.name, value)

class JEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)

def _build_xofields_dict(bases, data):
    if '_xofields' in data.keys():
        xofields = data['_xofields'].copy()
    elif any(map(lambda b: hasattr(b, '_xofields'), bases)):
        n_filled = 0
        for bb in bases:
            if hasattr(bb, '_xofields') and len(bb._xofields.keys()) > 0:
                n_filled += 1
                if n_filled > 1:
                    raise ValueError(
                        f'Multiple bases have _xofields: {bases}')
                xofields = bb._xofields.copy()
    else:
        xofields = {}

    for nn, tt in xofields.items():
        if isclass(tt) and issubclass(tt, HybridClass):
            xofields[nn] = tt._XoStruct

    return xofields


class MetaHybridClass(type):

    def __new__(cls, name, bases, data):

        if ('_xofields' not in data.keys()
                and any(map(lambda b: hasattr(b, '_XoStruct'), bases))):
            # No action, use _XoStruct from base class (used to build PyHEADTAIL interface)
            return type.__new__(cls, name, bases, data)

        _XoStruct_name = name + "Data"

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)

        _XoStruct = type(_XoStruct_name, (Struct,), xofields)

        if '_rename' in data.keys():
            rename = data['_rename']
        else:
            rename = {}

        new_class = type.__new__(cls, name, bases, data)

        new_class._XoStruct = _XoStruct

        new_class._rename = rename

        pynames_list = []
        for ff in _XoStruct._fields:
            fname = ff.name
            if fname in rename.keys():
                pyname = rename[fname]
            else:
                pyname = fname
            pynames_list.append(pyname)

            setattr(new_class, pyname, _FieldOfDressed(fname, _XoStruct))

            new_class._fields = pynames_list

        _XoStruct._DressingClass = new_class

        if '_extra_c_sources' in data.keys():
            new_class._XoStruct._extra_c_sources.extend(data['_extra_c_sources'])

        if '_depends_on' in data.keys():
            new_class._XoStruct._depends_on.extend(data['_depends_on'])

        if '_kernels' in data.keys():
            kernels = data['_kernels'].copy()
            for nn, kk in kernels.items():
                for aa in kk.args:
                    if aa.atype is ThisClass:
                        aa.atype = new_class._XoStruct
                    if isclass(aa.atype) and issubclass(aa.atype, HybridClass):
                        aa.atype = aa.atype._XoStruct
            new_class._XoStruct._kernels.update(kernels)

        for ii, tt in enumerate(new_class._XoStruct._depends_on):
            if isclass(tt) and issubclass(tt, HybridClass):
                new_class._XoStruct._depends_on[ii] = tt._XoStruct

        return new_class


class HybridClass(metaclass=MetaHybridClass):

    def move(self, _context=None, _buffer=None, _offset=None):
        self._xobject = self._xobject.__class__(
            self._xobject, _context=_context, _buffer=_buffer, _offset=_offset
        )
        self._reinit_from_xobject(_xobject=self._xobject)

    @property
    def _move_to(self):
        raise NameError("`_move_to` has been removed. Use `move` instead.")

    def _reinit_from_xobject(self, _xobject):
        self._xobject = _xobject
        for ff in self._XoStruct._fields:
            if hasattr(ff.ftype, "_DressingClass"):
                vv = ff.ftype._DressingClass(
                    _xobject=getattr(_xobject, ff.name)
                )
                pyname = self._rename.get(ff.name, ff.name)
                setattr(self, pyname, vv)

    def xoinitialize(self, _xobject=None, _kwargs_name_check=True, **kwargs):

        if _kwargs_name_check:
            fnames = [ff.name for ff in self._XoStruct._fields]
            for kk in kwargs.keys():
                if kk.startswith('_'):
                    continue
                if kk not in fnames:
                    raise NameError(
                        f'Invalid keyword argument `{kk}`')

        if _xobject is not None:
            self._reinit_from_xobject(_xobject=_xobject)
        else:
            # Handle dressed inputs
            dressed_kwargs = {}
            for kk, vv in kwargs.items():
                if hasattr(vv, "_xobject"):  # vv is dressed
                    dressed_kwargs[kk] = vv
                    kwargs[kk] = vv._xobject

            self._xobject = self._XoStruct(**kwargs)

            # Handle dressed inputs
            for kk, vv in dressed_kwargs.items():
                if kk in self._rename.keys():
                    pyname = self._rename[kk]
                else:
                    pyname = kk
                setattr(self, pyname, vv)

            # dress what can be dressed
            # (for example in case object is initialized from dict)
            self._reinit_from_xobject(_xobject=self._xobject)

    def __init__(self, _xobject=None, **kwargs):
        self.xoinitialize(_xobject=_xobject, **kwargs)

    def to_dict(self, copy_to_cpu=True):
        out = {"__class__": self.__class__.__name__}
        # Use a cpu copy by default:
        if copy_to_cpu:
            obj = self.copy(_context=context_default)
        else:
            obj = self

        for ff in obj._fields:
            if (hasattr(self, '_skip_in_to_dict')
                    and ff in self._skip_in_to_dict):
                continue
            vv = getattr(obj, ff)
            if hasattr(vv, "to_dict"):
                out[ff] = vv.to_dict()
            elif hasattr(vv, "_to_dict"):
                out[ff] = vv._to_dict()
            else:
                out[ff] = vv

        if hasattr(obj, '_store_in_to_dict'):
            for nn in obj._store_in_to_dict:
                out[nn] = getattr(obj, nn)

        return out

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, _offset=None):
        return cls(**dct, _context=_context, _buffer=_buffer, _offset=_offset,
                   _kwargs_name_check=False)

    def copy(self, _context=None, _buffer=None, _offset=None):
        if _context is None and _buffer is None:
            _context = self._xobject._buffer.context
        # This makes a copy of the xobject
        xobject = self._XoStruct(
            self._xobject, _context=_context, _buffer=_buffer, _offset=_offset
        )
        return self.__class__(_xobject=xobject)

    @property
    def _buffer(self):
        return self._xobject._buffer

    @property
    def _offset(self):
        return self._xobject._offset

    @property
    def _context(self):
        return self._xobject._buffer.context

    @property
    def XoStruct(self):
        raise NameError("`XoStruct` has been removed. Use `_XoStruct` instead.")

    @property
    def extra_sources(self):
        raise NameError("`extra_sources` has been removed. Use `_extra_c_sources` instead.")

    def compile_kernels(self, *args, **kwargs):
        return self._xobject.compile_kernels(*args, **kwargs)

class ThisClass: # Place holder
    pass
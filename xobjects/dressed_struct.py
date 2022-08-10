# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import json

import numpy as np
from .struct import Struct
from .typeutils import context_default


class _FieldOfDressed:
    def __init__(self, name, XoStruct):
        self.name = name
        self.isnplikearray = False

        fnames = [ff.name for ff in XoStruct._fields]
        if self.name in fnames:
            ftype = getattr(XoStruct, self.name).ftype
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

    return xofields


class MetaDressedStruct(type):

    def __new__(cls, name, bases, data):

        XoStruct_name = name + "Data"

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)

        XoStruct = type(XoStruct_name, (Struct,), xofields)

        if '_rename' in data.keys():
            rename = data['_rename']
        else:
            rename = {}

        new_class = type.__new__(cls, name, bases, data)

        new_class.XoStruct = XoStruct

        new_class._rename = rename

        pynames_list = []
        for ff in XoStruct._fields:
            fname = ff.name
            if fname in rename.keys():
                pyname = rename[fname]
            else:
                pyname = fname
            pynames_list.append(pyname)

            setattr(new_class, pyname, _FieldOfDressed(fname, XoStruct))

            new_class._fields = pynames_list

        XoStruct._DressingClass = new_class

        XoStruct.extra_sources = []
        if 'extra_sources' in data.keys():
            new_class.XoStruct.extra_sources.extend(data['extra_sources'])

        return new_class


class DressedStruct(metaclass=MetaDressedStruct):

    def _move_to(self, _context=None, _buffer=None, _offset=None):
        self._xobject = self._xobject.__class__(
            self._xobject, _context=_context, _buffer=_buffer, _offset=_offset
        )
        self._reinit_from_xobject(_xobject=self._xobject)

    def _reinit_from_xobject(self, _xobject):
        self._xobject = _xobject
        for ff in self.XoStruct._fields:
            if hasattr(ff.ftype, "_DressingClass"):
                vv = ff.ftype._DressingClass(
                    _xobject=getattr(_xobject, ff.name)
                )
                pyname = self._rename.get(ff.name, ff.name)
                setattr(self, pyname, vv)

    def xoinitialize(self, _xobject=None, _kwargs_name_check=True, **kwargs):

        if _kwargs_name_check:
            fnames = [ff.name for ff in self.XoStruct._fields]
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

            self._xobject = self.XoStruct(**kwargs)

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
        xobject = self.XoStruct(
            self._xobject, _context=_context, _buffer=_buffer, _offset=_offset
        )
        return self.__class__(_xobject=xobject)

    def compile_custom_kernels(self, only_if_needed=False,
                               save_source_as=None):
        context = self._buffer.context

        if only_if_needed:
            all_found = True
            for kk in self.XoStruct.custom_kernels.keys():
                if kk not in context.kernels.keys():
                    all_found = False
                    break
            if all_found:
                return

        context.add_kernels(
            sources=self.XoStruct.extra_sources,
            kernels=self.XoStruct.custom_kernels,
            extra_classes=[self.XoStruct],
            save_source_as=save_source_as,
        )

    @property
    def _buffer(self):
        return self._xobject._buffer

    @property
    def _offset(self):
        return self._xobject._offset

    @property
    def _context(self):
        return self._xobject._buffer.context
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


def dress(XoStruct, rename={}):

    if hasattr(XoStruct, "_DressingClass"):
        raise ValueError("A Struct cannot be dressed multiple times")

    DressedXStruct = type(
        "Dressed" + XoStruct.__name__, (), {"XoStruct": XoStruct}
    )

    DressedXStruct._rename = rename

    for ff in ["_buffer", "_offset"]:
        setattr(DressedXStruct, ff, _FieldOfDressed(ff, XoStruct))

    pynames_list = []
    for ff in XoStruct._fields:
        fname = ff.name
        if fname in rename.keys():
            pyname = rename[fname]
        else:
            pyname = fname
        pynames_list.append(pyname)

        setattr(DressedXStruct, pyname, _FieldOfDressed(fname, XoStruct))

        DressedXStruct._fields = pynames_list

    XoStruct._DressingClass = DressedXStruct

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

    def xoinitialize(self, _xobject=None, **kwargs):

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

    def myinit(self, _xobject=None, **kwargs):
        self.xoinitialize(_xobject=_xobject, **kwargs)

    def to_dict(self, copy_to_cpu=True):
        out = {"__class__": self.__class__.__name__}
        # Use a cpu copy by default:
        if copy_to_cpu:
            obj = self.copy(_context=context_default)
        else:
            obj = self

        for ff in obj._fields:
            vv = getattr(obj, ff)
            if hasattr(vv, "to_dict"):
                out[ff] = vv.to_dict()
            elif hasattr(vv, "_to_dict"):
                out[ff] = vv._to_dict()
            else:
                out[ff] = vv
        return out

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, _offset=None):
        return cls(**dct, _context=_context, _buffer=_buffer, _offset=_offset)

    def copy(self, _context=None, _buffer=None, _offset=None):
        if _context is None and _buffer is None:
            _context = self._xobject._buffer.context
        # This makes a copy of the xobject
        xobject = self.XoStruct(
            self._xobject, _context=_context, _buffer=_buffer, _offset=_offset
        )
        return self.__class__(_xobject=xobject)

    def compile_custom_kernels(self, only_if_needed=False):
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
            save_source_as="temp.c",
        )

    DressedXStruct.xoinitialize = xoinitialize
    DressedXStruct.compile_custom_kernels = compile_custom_kernels
    DressedXStruct.to_dict = to_dict
    DressedXStruct.from_dict = from_dict
    DressedXStruct.copy = copy
    DressedXStruct.__init__ = myinit
    DressedXStruct._reinit_from_xobject = _reinit_from_xobject
    DressedXStruct._move_to = _move_to

    return DressedXStruct


class JEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class MetaDressedStruct(type):
    def __new__(cls, name, bases, data):
        XoStruct_name = name + "Data"
        if "_xofields" in data.keys():
            xofields = data["_xofields"]
        else:
            for bb in bases:
                if hasattr(bb, "_xofields"):
                    xofields = bb._xofields
                    break
        XoStruct = type(XoStruct_name, (Struct,), xofields)

        bases = (dress(XoStruct),) + bases
        new_class = type.__new__(cls, name, bases, data)

        XoStruct._DressingClass = new_class

        return new_class


class DressedStruct(metaclass=MetaDressedStruct):
    _xofields = {}

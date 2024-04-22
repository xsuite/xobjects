# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import json
from inspect import isclass

import numpy as np
from .struct import Struct
from .typeutils import context_default
from .ref import Ref


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

            # Copy xobject data from value inside self._xobject
            # (unless same memory area or Ref and same buffer,
            #  in the latter case reference mechanism is used)
            if not (
                container._xobject._buffer is value._xobject._buffer
                and (
                    getattr(container._xobject, self.name) is not None
                    and getattr(container._xobject, self.name)._offset
                    == value._xobject._offset
                )
            ):
                setattr(container._xobject, self.name, value._xobject)

            if isinstance(getattr(container._XoStruct, self.name).ftype, Ref):
                if value._buffer is not container._buffer:
                    raise MemoryError(
                        "Cannot make a reference to an object in "
                        "a different buffer."
                    )
                # Reference mechanism was used
                setattr(container, "_dressed_" + self.name, value)
                value._movable = False
                return

            # Build a dressed version of the newly made copy
            dressed_new = value.__class__(
                _xobject=getattr(container._xobject, self.name)
            )
            setattr(container, "_dressed_" + self.name, dressed_new)

            dressed_new._movable = False

            # Copy the python data (changes also dressed_new._xobject)
            dressed_new.__dict__.update(value.__dict__)

            # Restore correct _xobject
            dressed_new._xobject = getattr(container._xobject, self.name)
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
    if "_xofields" in data.keys():
        xofields = data["_xofields"].copy()
    elif any(map(lambda b: hasattr(b, "_xofields"), bases)):
        n_filled = 0
        for bb in bases:
            if hasattr(bb, "_xofields") and len(bb._xofields.keys()) > 0:
                n_filled += 1
                if n_filled > 1:
                    raise ValueError(f"Multiple bases have _xofields: {bases}")
                xofields = bb._xofields.copy()
    else:
        xofields = {}

    for nn, tt in xofields.items():
        if isclass(tt) and issubclass(tt, HybridClass):
            xofields[nn] = tt._XoStruct

    return xofields


class MetaHybridClass(type):
    def __new__(cls, name, bases, data):
        if "_xofields" not in data.keys() and any(
            map(lambda b: hasattr(b, "_XoStruct"), bases)
        ):
            # No action, use _XoStruct from base class (used to build PyHEADTAIL interface)
            return type.__new__(cls, name, bases, data)

        _XoStruct_name = data.get("_cname", name + "Data")

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)

        _XoStruct = type(_XoStruct_name, (Struct,), xofields)

        if "_rename" in data.keys():
            rename = data["_rename"]
            if (set(rename.keys()) | set(xofields.keys())) & set(
                rename.values()
            ):
                raise ValueError(
                    "Cannot rename fields to names of other fields"
                )

            inverse_rename = {v: k for k, v in rename.items()}
            if len(rename.keys()) != len(inverse_rename.keys()):
                raise ValueError("Two fields are renamed to the same name")
        else:
            rename, inverse_rename = {}, {}

        xo_fnames = [ff.name for ff in _XoStruct._fields]
        py_fnames = xo_fnames.copy()
        for kk, vv in rename.items():
            py_fnames.remove(kk)
            py_fnames.append(vv)

        new_class = type.__new__(cls, name, bases, data)

        new_class._XoStruct = _XoStruct

        new_class._rename = rename
        new_class._inverse_rename = inverse_rename
        new_class._py_fnames = py_fnames
        new_class._xo_fnames = xo_fnames

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

        if "_extra_c_sources" in data.keys():
            new_class._XoStruct._extra_c_sources.extend(
                data["_extra_c_sources"]
            )

        if "_depends_on" in data.keys():
            new_class._XoStruct._depends_on.extend(data["_depends_on"])

        if "_kernels" in data.keys():
            kernels = data["_kernels"].copy()
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
    _movable = True
    _overridable = True
    _force_moveable = False

    def move(self, _context=None, _buffer=None, _offset=None):
        if not self._movable and not self._force_moveable:
            raise MemoryError(
                "This object cannot be moved, likely because it "
                "lives within another. Please, make a copy."
            )

        if self._xobject._has_refs and not self._force_moveable:
            raise MemoryError(
                "This object cannot be moved, as it contains "
                "references to other objects."
            )

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
                if hasattr(self, "_dressed_" + ff.name):
                    old_vv = getattr(self, "_dressed_" + ff.name)
                else:
                    old_vv = None

                vv = ff.ftype._DressingClass(
                    _xobject=getattr(_xobject, ff.name)
                )

                # preserve pure python attributes
                if old_vv is not None:
                    for kk in old_vv.__dict__.keys():
                        if kk not in vv.__dict__.keys():
                            vv.__dict__[kk] = old_vv.__dict__[kk]

                pyname = self._rename.get(ff.name, ff.name)
                setattr(self, pyname, vv)

    def xoinitialize(self, _xobject=None, _kwargs_name_check=True, **kwargs):
        if _kwargs_name_check:
            for kk in kwargs.keys():
                if kk.startswith("_"):
                    continue
                if kk not in self._py_fnames and kk not in self._xo_fnames:
                    raise NameError(f"Invalid keyword argument `{kk}`")

        if _xobject is not None:
            self._reinit_from_xobject(_xobject=_xobject)
            return

        # Handle dressed inputs
        dressed_kwargs, xo_kwargs = {}, {}
        for kk, vv in kwargs.items():
            if hasattr(vv, "_xobject"):  # vv is dressed
                dressed_kwargs[kk] = vv
                xo_kwargs[self._inverse_rename.get(kk, kk)] = vv._xobject
            else:
                xo_kwargs[self._inverse_rename.get(kk, kk)] = vv

        self._xobject = self._XoStruct(**xo_kwargs)

        # Handle dressed inputs
        for kk, vv in dressed_kwargs.items():
            setattr(self, kk, vv)

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

        skip_fields = set(getattr(obj, "_skip_in_to_dict", []))
        additional_fields = set(getattr(obj, "_store_in_to_dict", []))
        fields_to_store = (set(obj._fields) - skip_fields) | additional_fields

        defaults = {}
        for field in obj._XoStruct._fields:
            try:
                defaults[field.name] = field.get_default()
            except (TypeError, ValueError):
                # The above can fail with different error types
                # if a field type is dynamic.
                pass

        for ff in fields_to_store:
            vv = getattr(obj, ff)
            if hasattr(vv, "to_dict"):
                out[ff] = vv.to_dict()
            elif hasattr(vv, "_to_dict"):
                out[ff] = vv._to_dict()
            elif np.any(defaults.get(ff) != vv):
                # Only include those scalar values that are not default.
                out[ff] = vv

        return out

    @staticmethod
    def _static_from_dict(cls, dct, _context=None, _buffer=None, _offset=None):
        return cls(
            **dct,
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
            _kwargs_name_check=False,
        )

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, _offset=None):
        return HybridClass._static_from_dict(
            cls,
            dct,
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
        )

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

    def __getstate__(self):
        return self._xobject.__getstate__()

    def __setstate__(self, state):
        self._xobject = self._XoStruct._from_buffer(
            buffer=state[0], offset=state[1]
        )
        self._reinit_from_xobject(_xobject=self._xobject)

    @property
    def XoStruct(self):
        raise NameError(
            "`XoStruct` has been removed. Use `_XoStruct` instead."
        )

    @property
    def extra_sources(self):
        raise NameError(
            "`extra_sources` has been removed. Use `_extra_c_sources` instead."
        )

    def compile_kernels(self, *args, **kwargs):
        return self._xobject.compile_kernels(*args, **kwargs)

    def __repr__(self):

        if hasattr(self, "_repr_fields"):
            fnames = self._repr_fields
        else:
            fnames = []
            if hasattr(self, "_add_to_repr"):
                fnames += self._add_to_repr
            fnames += [fname for fname in self._fields]
            if hasattr(self, "_skip_in_repr"):
                fnames = [ff for ff in fnames if ff not in self._skip_in_repr]

        args = []
        for fname in fnames:
            vv = getattr(self, fname)
            if isinstance(vv, float):
                vvrepr = f"{vv:.3g}"
            else:
                vvrepr = repr(vv)
            args.append(f"{fname}={vvrepr}")
        return f'{type(self).__name__}({", ".join(args)})'


class ThisClass:  # Place holder
    pass

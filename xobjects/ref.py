import numpy as np

from .typeutils import Info
from xobjects import Int64

class MetaRef(type):

    def __getitem__(cls, rtype):
        return cls(rtype)

class Ref(metaclass=MetaRef):

    def __init__(self, rtype):
        if hasattr(rtype,' __iter__'):
            self._rtypes = rtype
            self._isunion = True
            self._size = 16
        else:
            self._rtype = rtype
            self._isunion = False
            self._size = 8

    def _typeid_from_type(self, typ):
        for ii, tt in self._rtypes:
            if typ is tt:
                return ii
        # If no match found:
        raise TypeError(f'{typ} not registered types!')

    def _type_from_typeid(self, typeid):
        for ii, tt in self._rtypes:
            if ii == typeid:
                return tt
        # If no match found:
        raise TypeError(f'Invalid id: {typeid}!')

    def _get_stored_type(self):
        typeid = Int64._from_buffer(buffer, offset + 8)
        return self._type_from_typeid(typeid)

    def _from_buffer(self, buffer, offset):
        refoffset = Int64._from_buffer(buffer, offset)
        if self._isunion:
            rtype = self._get_stored_type()
        else:
            rtype = self._rtype
        return rtype._from_buffer(buffer, refoffset)

    def _to_buffer(self, buffer, offset, value, info=None):

        # Get/set reference type
        if self._isunion:
            if (hasattr(value, '__class__')
                and value.__class__ in self._rtypes):
                rtype = value.__class__
                typeid = self._typeid_from_type(rtype)
                Int64._to_buffer(buffer, offset + 8, typeid)
            elif value is None:
                # Use the first type (default)
                rtype = self._rtypes[0]
                Int64._to_buffer(buffer, offset + 8, 0)
            else:
                # Keep old type
                rtype = self._get_stored_type()
        else:
            rtype = self._rtype

        # Get/set content
        if (isinstance(value, rtype)
                    and value._buffer is buffer):
                refoffset = value._offset
        else:
            if value is None:
                refoffset = -1
            else:
                newobj = rtype(value, _buffer=buffer)
                refoffset = newobj._offset
        Int64._to_buffer(buffer, offset, refoffset)

    def __call__(self, value=None):
        if value is None:
            return None
        else:
            raise NotImplementedError

    def _inspect_args(self, arg):
        return Info(size=self._size)

    #def __repr__(self):
    #    return self



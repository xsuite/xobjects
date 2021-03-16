"""

Current implementation:
    1) instance stores the actual offsets for dynamic offset. Although it is wasteful for memory, it avoids a double round trip. This hinges on the structure being frozen at initializations


Data layout:


Struct:
  [ instance size ]
  static-field1
  ..
  static-fieldn
  [ offset field 2 ]
  [ offset ...
  [ offset field n ]
  [ dynamic-field1 ]
  [ ...
  [ dynamic-fieldn ]


"""

from .context import get_a_buffer, dispatch_arg

from .scalar import Int64

import logging

log=logging.getLogger(__name__)


def _to_slot_size(size):
    "round to nearest multiple of 8"
    return (size + 7) & (-8)


def _is_dynamic(cls):
    return cls._size is None


class Field:
    def __init__(self, ftype, default=None, readonly=False, default_factory=None):
        self.ftype = ftype
        self.default=default
        if default_factory is not None:
            self.get_default=default_factory
        self.index = None  # filled by class creation
        self.name = None  # filled by class creation
        self.offset = None # filled by class creation
        self.deferenced = None # filled by class creation
        self.readonly = readonly

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        else:
            if self.deferenced:
                #offest cached in python instance to avoid double query
                offset = instance._offset + instance._offsets[self.index]
            else:
                offset = instance._offset + self.offset
            return self.ftype._from_buffer(instance._buffer, offset)

    def __set__(self, instance, value):
        """
        A value can be:
          - python value: dict or scalar or list...
          - another instance of the same type
          - ???a buffer_protocol object???
        """
        if self.readonly:
            raise AttributeError(f"Field {self.name} is read-only")
        if self.deferenced:
            #offest cached in python instance to avoid double query
            offset = instance._offset + instance._offsets[self.index]
        else:
            offset = instance._offset + self.offset
        self.ftype._to_buffer(instance._buffer, offset, value)


    def get_default(self):
        return dispatch_arg(self.ftype,self.default)


class MetaStruct(type):
    def __new__(cls, name, bases, data):
        offset = 0
        fields = []
        s_fields = []
        d_fields = []
        is_static = True
        findex=0
        for aname, field in data.items():
            if isinstance(field, Field):
                field.index = findex
                findex+=1
                field.name = aname
                fields.append(field)
                if field.ftype._size is None:
                    d_fields.append(field)
                    is_static = False
                else:
                    s_fields.append(field)
        data["_fields"] = fields
        data["_s_fields"] = s_fields
        data["_d_fields"] = d_fields

        if is_static:
            offset = 0
            for field in fields:
                field.offset=offset
                field.deferenced=False
                offset += _to_slot_size(field.ftype._size)
            size = offset

            def _get_size(self):
                return self.__class__._size

            def _get_size_from_args(cls, *args, **nargs):
                return cls._size, None

        else:
            size = None
            d_offsets = {} # offset of dynamic data (class invariant)
            for field in fields:
                offset = 8  # first slot is instance size
                for field in s_fields:
                    field.offset=offset
                    field.deferenced=False
                    offset += _to_slot_size(field.ftype._size)
                # other dynamic fields
                for field in d_fields[1:]:
                    field.offset=offset
                    field.deferenced=True
                    offset += _to_slot_size(8)
                # first dynamic field
                d_fields[0].offset=offset
                field.deferenced=False

            def _get_size(self):
                return Int64._from_buffer(self._buffer, self._offset)

            def _get_size_from_args(cls, *args, **nargs):
                log.debug(f"get size for {cls} from {args} {nargs}")
                if len(args)==1:
                    arg=args[0]
                    if isinstance(arg, dict):
                       offsets = {} # offset of dynamic data
                       extra =[]
                       offset = d_fields[0].offset # offset first dynamic data
                       log.debug(f"{arg}")
                       for field in cls._d_fields:
                         farg =  arg.get(field.name, field.get_default())
                         log.debug(f"get size for field `{field.name}` using `{farg}`")
                         fsize, foffsets =\
                                 dispatch_arg(field.ftype._get_size_from_args,farg)
                         log.debug(f"got size {fsize} and {foffsets}")
                         if foffsets is not None:
                            extra.append( (field.index,foffsets) )
                         offsets[field.index]= offset
                         offset += _to_slot_size(fsize)
                       size=offset
                       return size, (offsets, extra)
                    elif isinstance(arg, cls):
                        size=arg._get_size()
                        return size,None # extra info not needed
                    else:
                        raise ValueError(f"{arg} Not valid type for {cls}")
                else: #python argument
                    return cls._get_size_from_args(nargs)

        data["_size"] = size
        data["_get_size"] = _get_size
        data["_get_size_from_args"] = classmethod(_get_size_from_args)

        return type.__new__(cls, name, bases, data)


class Struct(metaclass=MetaStruct):
    @classmethod
    def _from_buffer(cls, buffer, offset):
        self = object.__new__(cls)
        self._buffer = buffer
        self._offset = offset
        _offsets={}
        for field in self._d_fields:
            offset=self._offset + field.offset
            val=Int64._from_buffer(self._buffer, offset)
            _offsets[field.index]=val
        self._offsets=_offsets
        return self

    @classmethod
    def _to_buffer(cls, buffer, offset, value, mdata=None):
        if isinstance(value, cls): #binary copy
            value_size=value._size
            if value._buffer.context is buffer.context:
                buffer.copy_from(value._buffer, value._offset, offset, value_size)
            else:
                data = value._buffer.read(value._offset, value_size)
                buffer.write(offset, data)
        else: # value must be a dict, again potential disctructive
            if mdata is None:
                size,mdata = cls._get_size_from_args(value)
            if mdata is not None:
                if len(mdata)==2:
                  loffsets,extra= mdata
                  cls._set_offsets(buffer, offset, loffsets)
            else:
                extra={}
            for field in cls._fields:
                fvalue = value.get(field.name, field.get_default())
                if field.deferenced:
                    foffset=offset+loffset[field.index]
                    if field.index in extra:
                       field.ftype._to_buffer(
                               buffer, foffset, fvalue, extra[field.index])
                else:
                    foffset=offset+field.offset
                    field.ftype._to_buffer( buffer, foffset, fvalue)

    def __init__(self, _context=None, _buffer=None, _offset=None,**nargs):
        """
        Create new struct in buffer from offset.
        If offset not provide 
        """
        cls = self.__class__
        # compute resources
        size, offsets = cls._get_size_from_args(**nargs)
        if offsets is not None:  # dynamic struct
            self._size = size
        # acquire buffer
        self._buffer, self._offset=get_a_buffer(size)
        # if dynamic struct store dynamic offsets
        if offsets is not None:
            self._offsets=offsets[0] #struct offsets
        self.__class__._to_buffer(self._buffer, self._offset, nargs, offsets)

    @classmethod
    def _set_offsets(cls, buffer, offset, loffsets):
        log.debug(f"{cls} set offset {loffsets}")
        for index, data_offset in loffsets.items():
            foffset=offset + cls._fields[index].offset
            Int64._to_buffer(buffer, foffset, data_offset)

    def _update_from_dict(self, data):
        for field in self._fields:
            if field.name in data:
               field.__set__(self, data[field.name])

    def _update_from_buffer(self, buffer, offset):
        # potentially destructive
        if self._buffer.context is buffer.context:
            self._buffer.copy_from(self._offset, offset, self._size)
        else:
            data = buffer.read(offset, self._size)
            self._buffer.write(self._offset, data)

    def _update_from_bytes(self, data):
        # potentially destructive
        self._buffer.write(self._offset, data[: self._size])

    def _to_dict(self):
        return {field.name: field.__get__(self) for field in self._fields}

    def __repr__(self):
        fields=((field.name, repr(getattr(self,field.name))) for field in self._fields)
        fields=', '.join(f"{k}={v}" for k,v in fields)
        longform=f"{self.__class__.__name__}({fields})"
        if len(longform)>60:
            return f"{self.__class__.__name__}(...)"
        else:
            return longform

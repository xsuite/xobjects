from .typeutils import Info
from .scalar import Int64
from .array import Array
from . import capi


class MetaRef(type):
    def __getitem__(cls, rtypes):
        if not isinstance(rtypes, tuple):
            rtypes = (rtypes,)
        return cls(*rtypes)


NULLVALUE = -(2 ** 63)


class Ref(metaclass=MetaRef):
    def __init__(self, *rtypes):
        self._rtypes = rtypes
        # self._rtypes_names = [tt.__name__ for tt in rtypes]
        self.__name__ = "Ref" + "".join(tt.__name__ for tt in self._rtypes)
        self._c_type = self.__name__

        if len(rtypes) == 1:
            self._is_union = False
            self._size = 8
        else:
            self._is_union = True
            self._size = 16

    def _typeid_from_type(self, typ):
        for ii, tt in enumerate(self._rtypes):
            if tt.__name__ == typ.__name__:
                return ii
        # If no match found:
        raise TypeError(f"{typ} is not memberof {self}!")

    def _typeid_from_name(self, name):
        for ii, tt in enumerate(self._rtypes):
            if tt.__name__ == name:
                return ii
        # If no match found:
        raise TypeError(f"{name} is not memberof {self}")

    def _type_from_typeid(self, typeid):
        for ii, tt in enumerate(self._rtypes):
            if ii == typeid:
                return tt
        # If no match found:
        raise TypeError(f"Invalid id: {typeid}!")

    def _is_member(self, rtype):
        for tt in self._rtypes:
            if rtype.__name__ == tt.__name__:
                return True
        return False

    def _type_from_name(self, name):
        for tt in self._rtypes:
            if tt.__name__ == name:
                return tt
        # If no match found:
        raise TypeError(f"Invalid name: {name}!")

    def _get_stored_type(self, buffer, offset):
        typeid = Int64._from_buffer(buffer, offset + 8)
        return self._type_from_typeid(typeid)

    def _from_buffer(self, buffer, offset):
        refoffset = Int64._from_buffer(buffer, offset)
        if refoffset == NULLVALUE:
            return None
        else:
            refoffset += offset
            if self._is_union:
                rtype = self._get_stored_type(buffer, refoffset)
            else:
                rtype = self._rtypes[0]
            return rtype._from_buffer(buffer, refoffset)

    def _to_buffer(self, buffer, offset, value, info=None):

        # Get/set reference type
        if self._is_union:
            if value is None:
                # Use the first type (default)
                rtype = -1  # self._rtypes[0]
                Int64._to_buffer(buffer, offset + 8, -1)
            elif self._is_member(value.__class__):
                rtype = value.__class__
                typeid = self._typeid_from_type(rtype)
                Int64._to_buffer(buffer, offset + 8, typeid)
            elif isinstance(value, tuple):
                rtype = self._typeid_from_name(value[0])
                Int64._to_buffer(buffer, offset + 8, typeid)
            else:
                # Keep old type
                rtype = self._get_stored_type(buffer, offset)
        else:
            rtype = self._rtypes[0]

        # Get/set content
        if value is None:
            refoffset = NULLVALUE  # NULL value
            Int64._to_buffer(buffer, offset, refoffset)
        elif (
            value.__class__.__name__ == rtype.__name__  # same type
            and value._buffer is buffer
        ):
            refoffset = value._offset - offset
            Int64._to_buffer(buffer, offset, refoffset)

        else:
            newobj = rtype(value, _buffer=buffer)
            refoffset = newobj._offset - offset
            Int64._to_buffer(buffer, offset, refoffset)

    def __call__(self, *args):
        if len(args) == 0:
            return None
        else:
            if self._is_union:
                name, value = args
                return self._type_from_name(name)(value)
            else:
                (value,) = args
                return self._rtypes[0](value)

    def _inspect_args(self, arg):
        return Info(size=self._size)

    def __getitem__(self, shape):
        return Array.mk_arrayclass(self, shape)

    def _get_c_offset(self, conf):
        itype = conf.get("itype", "int64_t")
        doffset = f"offset"  # starts of data
        return [f"  offset=(({itype}*) obj)[{doffset}];"]

    def _gen_data_paths(self, base=None):
        paths = []
        if base is None:
            base = []
        if self._is_union:
            paths.append(base + [self])
            for rtype in self._rtypes:
                if hasattr(rtype, "_gen_data_paths"):
                    paths.extend(rtype._gen_data_paths())
        else:
            rtype = self._rtypes[0]
            if hasattr(rtype, "_gen_data_paths"):
                paths.extend(rtype._gen_data_paths(base + [self]))
        return paths

    def _gen_c_api(self, conf={}):
        paths = self._gen_data_paths()
        return capi.gen_code(paths, conf)

    def __repr__(self):
        return f"<ref {self.__name__}>"


class MetaUnionRef(type):
    _rtypes: list

    def __getitem__(cls, shape):
        return Array.mk_arrayclass(cls, shape)

    @classmethod
    def _pre_init(cls, *arg, **kwargs):
        return kwargs

    def _post_init(self):
        pass

    def _from_buffer(cls, buffer, offset):
        refoffset, typeid = Int64._array_buffer(buffer, offset, 2)
        if refoffset == NULLVALUE:
            return None
        else:
            rtype = cls._type_from_typeind(typeid)
            return rtype._from_buffer(buffer, offset + refoffset)

    def _to_buffer(cls, buffer, offset, value, info=None):
        pass


class UnionRef(metaclass=MetaUnionRef):
    pass

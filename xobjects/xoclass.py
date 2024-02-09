# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
"""A structure type for Xobjects with dynamic method dispatch.

Apart from the regular behaviour expected from an Xobjects struct, one can
define C methods that will be exposed as kernels on the Python side, and on the
C side, through special methods, which allow the execution of the right method
without full knowledge of the type.

Example:

        class Base(xo.XoClass):
    >>>     res = xo.UInt64
    >>>     a = xo.UInt32
    >>>     b = xo.UInt32
    >>>     method_sum = Method(
    >>>         \"\"\"
    >>>         uint64_t Base_method_sum(Base this) {
    >>>             uint64_t sum = Base_get_a(this) + Base_get_b(this);
    >>>             Base_set_res(this, sum);
    >>>             return sum;
    >>>         }
    >>>         \"\"\",
    >>>         name='Base_method_sum',
    >>>         return_type=xo.Arg(xo.UInt64),
    >>>         args=[],
    >>>     )

    >>> class Child(Base):
    >>>     c = xo.UInt32
    >>>     method_sum = Method(
    >>>         \"\"\"
    >>>         uint64_t Child_method_sum(Child this) {
    >>>             uint64_t sum = Base_method_sum((Base)this) + Child_get_c(this);
    >>>             Child_set_res(this, sum);
    >>>             return sum;
    >>>         }
    >>>         \"\"\",
    >>>         name='Child_method_sum',
    >>>         return_type=xo.Arg(xo.UInt64),
    >>>         args=[],
    >>>     )

In the above example, Child inherits from Base, and overrides Base's method to
include the modification in the Child's body. We can then run:

    >>> child = Child(a=1, b=2, c=3)
    >>> child.compile_kernels()
    >>> child.method_sum()
    ... 6

On the C side, a function `uint64_t call_Base_method(Base obj)` is exposed, that
can be passed an opaque pointer to an instance of Base, or any of its
descendants, and which will dispatch the method to the correct implementation
based on the argument: if `obj` is a Base, to `Base_method`, and if `obj` is
really a Child, to `Child_method`.
"""
from typing import Type, List

from xobjects import UInt64
from xobjects.context import Source
from xobjects.struct import MetaStruct, Struct
from xobjects.methods import Method, MethodList
from xobjects.typeutils import default_conf


class MetaXoClass(MetaStruct):
    def __new__(mcs, name, bases, data):
        cls = super().__new__(mcs, name, bases, data)
        base_struct = cls._base_struct

        mcs._prepare_methods(cls, base_struct, data)

        for method_name, method in cls._methods.methods.items():
            cls._kernels[method.implementation_name] = method.get_func_kernel_description()

        if base_struct is None:
            cls._derived_classes = []
        elif base_struct is XoClass:
            cls._derived_classes = [cls]
        else:
            cls._derived_classes.append(cls)

        cls._class_type_id = len(cls._derived_classes)

        return cls

    @staticmethod
    def assert_inheritance_is_valid(data, base_struct):
        if base_struct is not None and base_struct == XoClass:
            return
        MetaStruct.assert_inheritance_is_valid(data, base_struct)

    @staticmethod
    def _prepare_methods(cls, base_struct, data):
        try:
            methods = base_struct._methods.methods.copy()
        except AttributeError:
            methods = {}

        for field_name, field in data.items():
            if isinstance(field, Method):
                field.set_method_name(field_name)
                field.set_this_class(cls)
                if field_name in methods:
                    field = methods[field_name].override(field, cls)
                methods[field_name] = field

        cls._methods = MethodList(implementor_class=cls, methods=methods)


class XoClass(Struct, metaclass=MetaXoClass):
    _type_id = UInt64

    _is_static = False
    _methods: MethodList = []
    _derived_classes: List[Type['XoClass']]

    def __init__(self, *args, **kwargs):
        Struct.__init__(self, *args, **kwargs)
        self._type_id = type(self)._class_type_id

    @classmethod
    def _generate_c_declaration(cls: Type['XoClass']):
        return cls._methods.get_c_declaration()

    @classmethod
    def _generate_c_implementation(cls: Type['XoClass']):
        return cls._methods.get_c_implementation()

    def compile_kernels(
        self,
        only_if_needed=False,
        apply_to_source=(),
        save_source_as=None,
        extra_classes=(),
    ):
        super().compile_kernels(
            only_if_needed, apply_to_source, save_source_as, extra_classes,
        )

    @classmethod
    def _gen_c_api(cls, conf=default_conf) -> Source:
        source = super()._gen_c_api(conf).source
        source += '\n\n'
        source += cls._generate_c_implementation()

        source = Source(
            source=source,
            name=cls.__name__ + "_gen_c_api",
        )
        return source

    @classmethod
    def _gen_c_decl(cls, conf=default_conf) -> str:
        source = super()._gen_c_decl(conf)
        source += '\n\n'
        source += cls._generate_c_declaration()
        return source
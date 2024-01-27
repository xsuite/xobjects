# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
"""A structure type for Xobjects with dynamic method dispatch."""
from typing import Type, List

from xobjects import UInt64
from xobjects.context import Source
from xobjects.struct import MetaStruct, Struct, Field
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
        source = f'// START {cls}._gen_c_api\n'
        source += super()._gen_c_api(conf).source
        source += '\n\n'
        source += cls._generate_c_implementation()
        source += f'// END {cls}._gen_c_api\n'

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
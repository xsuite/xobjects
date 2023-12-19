# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
from typing import List, Dict, Type

from xobjects.context import Arg, Kernel
from xobjects.base_type import XoType


class Method:
    def __init__(self, source: str, name: str, return_type: 'xo.Arg', args: List['xo.Arg']):
        self.source = source
        self.implementation_name = name
        self.return_type = return_type
        self.self_arg = Arg(atype=None, name='this')
        self.args = [self.self_arg] + args
        self.struct_instance = None
        self.method_name = None
        self.class_name = None

    def set_method_name(self, name):
        self.method_name = name

    def set_this_class(self, this_class):
        self.self_arg.atype = this_class
        self.class_name = this_class._c_type

    def get_c_declaration(self):
        c_return = self.return_type.get_c_type()
        c_args = [arg.get_c_type() for arg in self.args]
        c_arg_string = ', '.join(c_args)
        return f"{c_return} {self.implementation_name}({c_arg_string});"

    def get_c_implementation(self):
        return self.source

    def get_pointer(self, pointer_name):
        c_return = self.return_type.get_c_type()
        c_args = [arg.get_c_type() for arg in self.args]
        c_arg_string = ', '.join(c_args)
        return f"{c_return} (*{pointer_name})({c_arg_string})"

    def bind_to_instance(self, struct_instance: XoType):
        self.struct_instance = struct_instance

    def get_kernel_description(self):
        return Kernel(
            args=self.args,
            ret=self.return_type,
            c_name=self.implementation_name,
        )

    def __call__(self, **kwargs):
        if not self.struct_instance:
            raise RuntimeError(
                'The method is not bound to an xobject. Are you calling this '
                'method on an instance of an xobject?'
            )

        kernel = getattr(self.struct_instance._context.kernels, self.method_name)
        return kernel(this=self.struct_instance, **kwargs)


def method_from_source(source, name, return_type, args):
    return Method(source, name, return_type, args)


class MethodTable:
    def __init__(self, class_name: str, methods: Dict[str, Method]):
        self.class_name = class_name
        self.c_type_name = self.class_name + "MethodTable"
        self.c_name = self.class_name + "_method_table"
        self.methods = methods

    def get_c_declaration(self):
        lines = []

        # Generate method declarations
        for method in self.methods.values():
            lines.append(method.get_c_declaration() + '\n\n')

        # Generate method table typedef
        lines.append("typedef struct {")
        for method_name, method in self.methods.items():
            lines.append(method.get_pointer(method_name) + ";")
        lines.append(f"}} {self.c_type_name};")

        return '\n'.join(lines)

    def get_c_implementation(self):
        lines = []

        for method in self.methods.values():
            lines.append(method.get_c_implementation())

        lines.append(self._get_c_table_implementation())

        return '\n'.join(lines)

    def _get_c_table_implementation(self):
        initializer_list = []
        for method_name, method in self.methods.items():
            initializer_list.append(f".{method_name} = {method.implementation_name}")

        initializer_list_str = ', '.join(initializer_list)
        return f"{self.c_type_name} {self.c_name} = {{ {initializer_list_str} }};"

    def bind_to_instance(self, instance: XoType):
        for method in self.methods.values():
            method.bind_to_instance(instance)

    def bind_to_class(self, this_class: Type[XoType]):
        for method in self.methods.values():
            method.set_this_class(this_class)

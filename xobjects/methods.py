# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
from typing import List, Dict, Type

from xobjects.context import Arg, Kernel
from xobjects.base_type import XoType


class Method:
    """Create a method to be bound to an Xobjects class.

    Arguments
    ---------
    source
        The source code of the method.
    name
        The name of the method-kernel as in the `source`.
    args
        The arguments to the method-kernel as a list of `Arg` objects.
    return_type
        The return type of the method-kernel as an `Arg` object.
    """

    def __init__(self, source: str, name: str, return_type: Arg, args: List[Arg] = None):
        self.source = source
        self.implementation_name = name
        self.return_type = return_type
        self.self_arg = Arg(atype=None, name='this')
        self.args = [self.self_arg] + (args or [])
        self.method_name = None
        self.class_name = None
        self.implementors: 'XoClass' = []

    def set_method_name(self, name):
        """Set the name of the method.

        This is not the same as `implementation_name`, which is the name of
        the function implementing the behaviour of the method for this Xobject
        class, but rather the name of the method for this type: e.g. the method
        named 'do_something' could be implemented as 'SomeXoClass_do_something'
        for a class 'SomeXoClass', which could inherit from 'SomeBaseXoClass',
        which also has a method 'do_something', implemented as
        'SomeBaseXoClass_do_something'.
        """
        self.method_name = name

    def set_this_class(self, this_class: 'XoClass'):
        self.self_arg.atype = this_class
        self.class_name = this_class._c_type

    @property
    def this_class(self):
        return self.self_arg.atype

    def override(self, new_method: 'Method', implementor_class: 'XoClass'):
        self.register_with_class(implementor_class)
        new_method.implementors = self.implementors
        return new_method

    def get_c_declaration(self):
        """Generate a string with the declaration of this method."""
        c_return = self.return_type.get_c_type()
        c_args = [arg.get_c_type() for arg in self.args]
        c_arg_string = ', '.join(c_args)
        return f"{c_return} {self.implementation_name}({c_arg_string});"

    def get_arglist(self):
        c_args = [arg.get_c_type() + " " + arg.name for arg in self.args]
        return ', '.join(c_args)

    def get_c_implementation(self):
        """Print a string with the implementation of this method."""
        return self.source

    def register_with_class(self, implementor_class: 'XoClass'):
        if implementor_class not in self.implementors:
            self.implementors.append(implementor_class)

    def get_dispatcher_implem(self):
        """Print an implementation of a method that accepts `class_name` C type
        and dispatches it according to the correct actual type (based on type
        id code).

        The function looks as follows:

            RETTYPE call_Base_method_sum(Base this) {
                switch (Base_get__type_id(this))
                {
                    #ifdef XOBJ_TYPEDEF_Base
                    case 1: return Base_method_sum((Base)this); break;
                    #endif
                    #ifdef XOBJ_TYPEDEF_Child
                    case 2: return Child_method_sum((Child)this); break;
                    #endif
                }
                return (RETTYPE)0;
            }
        """
        dispatcher_decl = self.get_dispatcher_declaration()
        call_arg_list = ','.join([arg.name for arg in self.args])
        source = dispatcher_decl + (
            " {\n"
            f"switch ({self.class_name}_get__type_id({self.self_arg.name}))\n"
            "   {\n"
        )

        for current_class in self.implementors:
            type_id = current_class._class_type_id
            func = current_class._methods.get_implementation(self.method_name)
            func_name = func.implementation_name
            cast_to = func.this_class._c_type
            action = f"return {func_name}(({cast_to}){call_arg_list})"
            source += f"    #ifdef XOBJ_TYPEDEF_{current_class._c_type}\n"
            source += f"    case {type_id}: {action}; break;\n"
            source += f"    #endif\n"

        source += (
            "   }\n"
            f"   printf(\n"
            f"      \"Type code %llu unknown for method %s\", "
            f"      {self.class_name}_get__type_id({self.self_arg.name}), "
            f"      \"{self.dispatcher_name}\""
            f");\n"
            f"return ({self.return_type.get_c_type()})0;\n"
            "}\n"
        )

        return source

    def get_dispatcher_declaration(self):
        """Generate a string with the declaration of this method."""
        ret_type = self.return_type.get_c_type()
        return f"{ret_type} {self.dispatcher_name}({self.get_arglist()})"

    @property
    def dispatcher_name(self):
        first_implementor = self.implementors[0]._c_type
        return f'call_{first_implementor}_{self.method_name}'

    def get_pointer(self, pointer_name):
        """Generate a string with the C pointer to this method type."""
        c_return = self.return_type.get_c_type()
        c_args = [arg.get_c_type() for arg in self.args]
        c_arg_string = ', '.join(c_args)
        return f"{c_return} (*{pointer_name})({c_arg_string})"

    def get_func_kernel_description(self):
        """Generate a kernel description of the method for CFFI."""
        return Kernel(
            args=self.args,
            ret=self.return_type,
            c_name=self.implementation_name,
        )

    def __get__(self, instance, owner):
        if instance is None:
            raise ValueError('Cannot call an Xobjects method on the class '
                             'itself, only on an instance of an XoClass.')

        kernel = getattr(
            instance._context.kernels,
            self.implementation_name,
        )
        return lambda **kwargs: kernel(this=instance, **kwargs)

    def __repr__(self):
        return (
            f'Method('
            f'this_class={self.this_class}, '
            f'name={self.method_name}, '
            f'implementation_name={self.implementation_name})'
        )

    def is_last_implementor(self, implementor_class):
        type_id = implementor_class._class_type_id
        max_type_id = len(implementor_class._derived_classes) - 1
        return type_id == max_type_id


class MethodList:
    def __init__(self, implementor_class: 'XoClass', methods: Dict[str, Method]):
        self.implementor_class = implementor_class
        self.methods = methods

        for method in methods.values():
            method.register_with_class(implementor_class)

    def get_c_declaration(self):
        lines = []

        for method in self.methods.values():
            if method.this_class != self.implementor_class:
                # We want only do declare the methods defined in the current
                # class, otherwise they are inherited, and we do not wish to
                # repeat the declaration
                continue
            lines.append(method.get_c_declaration() + '\n\n')

        return '\n'.join(lines)

    def get_c_implementation(self):
        lines = []

        for method in self.methods.values():
            assert method.this_class is not None
            if method.this_class != self.implementor_class:
                # See comment in `self.get_c_declaration`
                continue
            lines.append(method.get_c_implementation())

            if method.is_last_implementor(self.implementor_class):
                lines.append(method.get_dispatcher_implem())

        return '\n'.join(lines)

    def get_implementation(self, method_name) -> Method:
        return self.methods[method_name]

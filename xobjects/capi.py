# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from .context import Kernel, Arg

from .scalar import UInt32, Int64, Void, is_scalar
from .struct import is_field, is_struct
from .array import is_index, is_array
from .ref import is_unionref, is_ref
from .string import is_string


def is_compound(xotype):
    """Return True for types passed in C as opaque object pointers.

    Parameters:
        xotype: xobjects type to check.

    Returns:
        True if the type is a struct, array, or unionref.
    """
    return is_struct(xotype) or is_array(xotype) or is_unionref(xotype)


def is_type(xotype):
    """Return True for any xobjects type with a C API representation.

    Parameters:
        xotype: xobjects type to check.

    Returns:
        True if the type can appear in generated C API methods.
    """
    return is_compound(xotype) or is_scalar(xotype) or is_string(xotype)


def get_inner_type(part):
    """Return the contained type for a field, array, or ref.

    Parameters:
        part: Path item, such as a field, array, ref, or unionref.

    Returns:
        Inner xobjects type, or None for a unionref.
    """
    if is_field(part):
        return part.ftype
    elif is_array(part):
        return part._itemtype
    elif is_ref(part):
        return part._reftype
    elif is_unionref(part):
        return None
    else:
        raise ValueError(f"Cannot get inner type of {part}")


def gen_pointer(ctype, conf):
    """Add the configured GPU memory qualifier to a pointer type.

    Parameters:
        ctype: C pointer type text.
        conf: C API generation options.

    Returns:
        Pointer type text with the configured memory qualifier.
    """
    gpu_memory_qualifier = conf.get("gpumem", "")
    return f"{gpu_memory_qualifier}{ctype}"


def gen_c_type_from_arg(arg: Arg, conf):
    """Return the C type used for an argument or return value.

    Parameters:
        arg: Kernel argument, return argument, or None for void.
        conf: C API generation options.

    Returns:
        C type text without an argument name.
    """
    gpu_memory_qualifier = conf.get("gpumem", "")
    if arg is None:
        c_type = "void"
    else:
        c_type = arg.atype._c_type
        if c_type.endswith("*"):  # handle literal char*
            c_type = f"{gpu_memory_qualifier}{c_type}"
        if arg.pointer:
            c_type = f"{gpu_memory_qualifier}{c_type}*"
        if arg.const:
            c_type = "const " + c_type
    return c_type


def gen_c_arg_from_arg(arg: Arg, conf):
    """Return one C function-argument declaration.

    Parameters:
        arg: Kernel argument to declare.
        conf: C API generation options.

    Returns:
        C argument declaration including the argument name.
    """
    gpu_memory_qualifier = conf.get("gpumem", "")
    cpu_restrict_qualifier = conf.get("cpurestrict", "")
    if arg is None:
        c_type = "void"
    else:
        c_type = arg.atype._c_type
        if arg.pointer:
            c_type = f"{gpu_memory_qualifier}{c_type}*{cpu_restrict_qualifier}"
        elif is_compound(arg.atype):
            c_type = f"{c_type}{cpu_restrict_qualifier}"
        if arg.const:
            c_type = "const " + c_type
    return f"{c_type} {arg.name}"


def gen_c_size_from_arg(arg: Arg, conf):
    """Return the C-side storage size, in bytes, for an argument.

    Parameters:
        arg: Kernel argument, return argument, or None for void.
        conf: C API generation options.

    Returns:
        Size in bytes, or None for void.
    """
    if arg is None:
        return None
    else:
        if is_compound(arg.atype):
            return conf.get("pointersize", 8)
        else:
            return arg.atype._size


def gen_c_decl_from_kernel(kernel: Kernel, conf):
    """Return a C function declaration for a kernel.

    Parameters:
        kernel: Kernel description to declare.
        conf: C API generation options.

    Returns:
        C function declaration without a trailing semicolon.
    """
    c_args = ", ".join([gen_c_arg_from_arg(arg, conf) for arg in kernel.args])
    if kernel.ret is None:
        return_type = "void"
    else:
        return_type = gen_c_type_from_arg(kernel.ret, conf)
    gpu_function_qualifier = conf.get("gpufun")
    if gpu_function_qualifier is None:
        return f"{return_type} {kernel.c_name}({c_args})"
    else:
        return (
            f"{gpu_function_qualifier} {return_type} {kernel.c_name}({c_args})"
        )


def get_layers(parts):
    """Count array-index layers in a path.

    Parameters:
        parts: Path items to inspect.

    Returns:
        Number of path items that have a shape.
    """
    layers = 0
    for part in parts:
        if hasattr(part, "_shape"):
            layers += 1
    return layers


def int_from_obj(offset, conf):
    """Generate code to read the integer at obj + offset.

    Parameters:
        offset: C expression for the byte offset from obj.
        conf: C API generation options.

    Returns:
        C expression that loads the configured integer type.
    """
    int_pointer_type = gen_pointer(conf.get("inttype", "int64_t") + "*", conf)
    char_pointer_type = gen_pointer(conf.get("chartype", "char") + "*", conf)
    return f"*({int_pointer_type})(({char_pointer_type}) obj+{offset})"


def Field_get_c_offset(self, conf):
    """Return C code or an integer offset for a field access.

    Parameters:
        self: Field object from an access path.
        conf: C API generation options.

    Returns:
        Static byte offset, or C lines that follow a dynamic reference.
    """
    if self.is_reference:
        data_offset_expr = f"offset+{self.offset}"
        reference_offset_expr = int_from_obj(data_offset_expr, conf)
        return [f"  offset+={reference_offset_expr};"]
    else:
        return self.offset


def Ref_get_c_offset(_ref, conf):
    """Return C code that follows a relative reference offset.

    Parameters:
        _ref: Ref object from an access path. Not used.
        conf: C API generation options.

    Returns:
        C lines that add the referenced object's relative offset.
    """
    reference_offset_expr = int_from_obj("offset", conf)
    return [f"  offset+={reference_offset_expr};"]


def Index_get_c_offset(index_part, conf, index_start):
    """Return C code that applies an array index to the byte offset.

    Parameters:
        index_part: Index object from an access path.
        conf: C API generation options.
        index_start: First C index argument used by this path item.

    Returns:
        C lines that update offset for this index.
    """
    array_cls = index_part.cls
    int_type = conf.get("inttype", "int64_t")

    code_lines = []
    if hasattr(array_cls, "_strides"):  # static shape or 1d dynamic shape
        strides = array_cls._strides
    else:
        ndim = len(array_cls._shape)
        strides = []
        for axis in range(ndim):
            stride_offset = 8 + (len(array_cls._dshape_idx) * 8) + (axis * 8)
            stride_name = f"{array_cls.__name__}_s{axis}"
            stride_value = int_from_obj(f"offset+{stride_offset}", conf)
            code_lines.append(f"  {int_type} {stride_name}={stride_value};")
            strides.append(stride_name)

    slot_offset = "+".join(
        f"i{axis + index_start}*{stride}"
        for axis, stride in enumerate(strides)
    )
    if array_cls._data_offset > 0:
        slot_offset = f"{array_cls._data_offset}+{slot_offset}"
    if array_cls._is_static_type:
        code_lines.append(f"  offset+={slot_offset};")
    else:
        lookup_field_offset = f"offset+{slot_offset}"
        code_lines.append(
            f"  offset+={int_from_obj(lookup_field_offset, conf)};"
        )
    return code_lines


def gen_method_offset(path, conf):
    """Return C code that computes the target byte offset.

    Parameters:
        path: Access path from root object to target.
        conf: C API generation options.

    Returns:
        C statements that initialize and update the offset variable.
    """
    int_type = conf.get("inttype", "int64_t")
    code_lines = [f"  {int_type} offset=0;"]
    pending_static_offset = 0
    next_index_arg = 0
    for part in path:
        if is_index(part):
            offset_step = Index_get_c_offset(part, conf, next_index_arg)
            next_index_arg += len(part.cls._shape)
        elif is_field(part):
            offset_step = Field_get_c_offset(part, conf)
        elif is_ref(part):
            offset_step = Ref_get_c_offset(part, conf)
        else:
            offset_step = None

        if type(offset_step) is int:
            pending_static_offset += offset_step
        elif type(offset_step) is list:
            if pending_static_offset > 0:
                code_lines.append(f"  offset+={pending_static_offset};")
            code_lines.extend(offset_step)
            pending_static_offset = 0
    if pending_static_offset > 0:
        code_lines.append(f"  offset+={pending_static_offset};")
    return "\n".join(code_lines)


def gen_fun_kernel(cls, path, action, const, extra, ret, add_nindex=False):
    """Build the Kernel object for one generated C API method.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to target.
        action: Method action name, such as get or shape.
        const: Whether the root object argument is const.
        extra: Extra Kernel arguments after obj and indices.
        ret: Kernel return Arg, or None for void.
        add_nindex: Append the number of index arguments to action.

    Returns:
        Kernel description for the generated C method.
    """
    type_name = cls._c_type
    field_names = []
    index_count = 0

    for part in path:
        if is_field(part):
            field_names.append(part.name)
        elif is_index(part):
            index_count += len(part.cls._shape)

    if add_nindex and index_count > 0:
        action += str(index_count)

    function_name_parts = [type_name, action]
    if len(field_names) > 0:
        function_name_parts.append("_".join(field_names))

    function_name = "_".join(function_name_parts)
    args = [Arg(cls, pointer=False, const=const, name="obj")]
    for index_arg in range(index_count):
        args.append(Arg(Int64, name=f"i{index_arg}"))

    args.extend(extra)

    return Kernel(args, c_name=function_name, ret=ret)


def gen_c_pointed(target: Arg, conf):
    """Return C code that reads or points to the target at obj + offset.

    Parameters:
        target: Argument describing the target type.
        conf: C API generation options.

    Returns:
        C expression usable as an lvalue or return value.
    """
    target_size = gen_c_size_from_arg(target, conf)
    target_c_type = gen_c_type_from_arg(target, conf)

    if target.pointer or is_compound(target.atype) or is_string(target.atype):
        char_pointer_type = gen_pointer(
            conf.get("chartype", "char") + "*", conf
        )
        return f"({target_c_type})(({char_pointer_type}) obj+offset)"

    target_pointer_type = gen_pointer(target_c_type + "*", conf)
    if target_size == 1:
        return f"*(({target_pointer_type}) obj+offset)"

    char_pointer_type = gen_pointer(conf.get("chartype", "char") + "*", conf)
    return f"*({target_pointer_type})(({char_pointer_type}) obj+offset)"


def gen_method_get(cls, path, conf):
    """Generate a C getter for a scalar path target.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to scalar target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    target_type = path[-1]
    return_arg = Arg(target_type)
    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="get",
        extra=[],
        ret=return_arg,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]
    code_lines.append(gen_method_offset(path, conf))
    target_expr = gen_c_pointed(return_arg, conf)
    code_lines.append(f"  return {target_expr};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_set(cls, path, conf):
    """Generate a C setter for a scalar path target.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to scalar target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    target_type = path[-1]
    value_arg = Arg(target_type, name="value")
    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="set",
        extra=[value_arg],
        ret=None,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]
    code_lines.append(gen_method_offset(path, conf))
    target_expr = gen_c_pointed(value_arg, conf)
    code_lines.append(f"  {target_expr}=value;")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_getp(cls, path, conf):
    """Generate a C getter that returns a pointer to the path target.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    target_type = path[-1]
    if target_type is None:
        return_arg = Arg(Void, pointer="True")
    else:
        return_arg = Arg(target_type)
    #    if is_scalar(target_type) or is_string(target_type):
    if is_scalar(target_type):
        return_arg.pointer = True

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="getp",
        extra=[],
        ret=return_arg,
        add_nindex=True,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]
    code_lines.append(gen_method_offset(path, conf))
    target_expr = gen_c_pointed(return_arg, conf)
    code_lines.append(f"  return {target_expr};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_len(cls, path, conf):
    """Generate a C method returning the total array length.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to array target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    array_type = path[-1]

    return_arg = Arg(Int64)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="len",
        extra=[],
        ret=return_arg,
        add_nindex=True,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    if array_type._is_static_shape:
        item_count = array_type._get_n_items()
        code_lines.append(f"  return {item_count};")
    else:
        code_lines.append(gen_method_offset(path, conf))
        header_arg = Arg(Int64, pointer=True)
        header_expr = gen_c_pointed(header_arg, conf)
        header_pointer_type = gen_pointer("int64_t*", conf)
        code_lines.append(f"  {header_pointer_type} arr = {header_expr};")
        dim_len_idx = 1
        terms = []
        for dim_len in array_type._shape:
            if dim_len:
                terms.append(str(dim_len))
            else:
                terms.append(f"arr[{dim_len_idx}]")
                dim_len_idx += 1
        length_expr = "*".join(terms)
        code_lines.append(f"  return {length_expr};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_size(cls, path, conf):
    """Generate a C method returning item size in bytes.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to container target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and Kernel, or (None, None) if size is unknown.
    """
    inner_type = get_inner_type(path[-1])
    if inner_type is None:  # cannot determine size
        return (None, None)
    return_arg = Arg(Int64)

    action = "size"
    layers = get_layers(path)
    if layers > 0 and not is_scalar(inner_type):
        action += str(layers)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action=action,
        extra=[],
        ret=return_arg,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    if inner_type._size is None:
        code_lines.append(gen_method_offset(path, conf))
        target_expr = gen_c_pointed(return_arg, conf)
        code_lines.append(f"  return {target_expr};")
    else:
        code_lines.append(f"  return {inner_type._size};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_shape(cls, path, conf):
    """Generate a C method that writes array shape into out_shape.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to array target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    array_type = path[-1]

    out_shape_arg = Arg(Int64, name="out_shape", pointer=True)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="shape",
        extra=[out_shape_arg],
        ret=None,
        add_nindex=True,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    ndim = len(array_type._shape)

    if array_type._is_static_shape:
        terms = [str(dim) for dim in array_type._shape]
    else:
        code_lines.append(gen_method_offset(path, conf))
        header_arg = Arg(Int64, pointer=True)
        header_expr = gen_c_pointed(header_arg, conf)
        header_pointer_type = gen_pointer("int64_t*", conf)
        code_lines.append(f"  {header_pointer_type} arr = {header_expr};")
        dim_len_idx = 1
        terms = []
        for dim_len in array_type._shape:
            if dim_len:
                terms.append(str(dim_len))
            else:
                terms.append(f"arr[{dim_len_idx}]")
                dim_len_idx += 1

    for dim_idx in range(ndim):
        code_lines.append(f"  out_shape[{dim_idx}] = {terms[dim_idx]};")

    code_lines.append("}")

    return "\n".join(code_lines), kernel


def gen_method_nd(cls, path, conf):
    """Generate a C method returning the number of array dimensions.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to array target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    array_type = path[-1]

    return_arg = Arg(UInt32)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="nd",
        extra=[],
        ret=return_arg,
        add_nindex=True,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    ndim = len(array_type._shape)
    code_lines.append(f"  return {ndim};")
    code_lines.append("}")

    return "\n".join(code_lines), kernel


def gen_method_strides(cls, path, conf):
    """Placeholder for a future C method returning array strides.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to array target.
        conf: C API generation options.

    Returns:
        Always (None, None); this method is not implemented.
    """
    return None, None


def gen_method_getpos(cls, path, conf):
    """Placeholder for a future C method returning a slot position.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to array target.
        conf: C API generation options.

    Returns:
        Always (None, None); this method is not implemented.
    """
    return None, None


def gen_method_typeid(cls, path, conf):
    """Generate a C method returning a unionref type id.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to unionref target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    return_arg = Arg(Int64)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="typeid",
        extra=[],
        ret=return_arg,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    code_lines.append(gen_method_offset(path, conf))
    target_expr = gen_c_pointed(return_arg, conf)
    code_lines.append("  offset+=8;")
    code_lines.append(f"  return {target_expr};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_member(cls, path, conf):
    """Generate a C method returning the active unionref member.

    Parameters:
        cls: Root xobjects class for the generated method.
        path: Access path from root object to unionref target.
        conf: C API generation options.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    return_arg = Arg(Void, pointer=True)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="member",
        extra=[],
        ret=return_arg,
    )
    declaration = gen_c_decl_from_kernel(kernel, conf)

    code_lines = [declaration + "{"]

    code_lines.append(gen_method_offset(path, conf))

    code_lines.extend(Ref_get_c_offset("offset", conf))
    target_expr = gen_c_pointed(return_arg, conf)
    code_lines.append(f" return {target_expr};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_method_switch(cls, path, conf, method):
    """Generate a C dispatcher for a unionref method.

    Parameters:
        cls: Root unionref class for the generated method.
        path: Access path ending at that unionref type.
        conf: C API generation options.
        method: Method descriptor declared on the unionref.

    Returns:
        Tuple of C source text and its Kernel description.
    """
    unionref_type = path[-1]

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action=method.c_name,
        extra=method.args,
        ret=method.ret,
    )
    unionref_name = unionref_type.__name__

    declaration = gen_c_decl_from_kernel(kernel, conf)
    code_lines = [declaration + "{"]
    void_pointer_type = gen_pointer("void*", conf)
    code_lines.append(
        f"  {void_pointer_type} member = {unionref_name}_member(obj);"
    )
    code_lines.append(f"  switch ({unionref_name}_typeid(obj)){{")
    for member_type in unionref_type._reftypes:
        member_type_name = member_type.__name__
        call_args = [f"({member_type_name}) member"]
        for arg in kernel.args[1:]:
            call_args.append(f"{arg.name}")
        call_args = ",".join(call_args)
        code_lines.append(f"""\
        #ifndef {unionref_name.upper()}_SKIP_{member_type_name.upper()}
        case {unionref_name}_{member_type_name}_t:
            return {member_type_name}_{method.c_name}({call_args});
            break;
        #endif""")
    code_lines.append("  }")
    code_lines.append(f"  return{'' if method.ret is None else ' 0'};")
    code_lines.append("}")
    return "\n".join(code_lines), kernel


def gen_typedef(cls, conf):
    """Generate the opaque C typedef for a compound type.

    Parameters:
        cls: xobjects class to typedef.
        conf: C API generation options.

    Returns:
        C typedef statement.
    """
    gpu_memory_qualifier = conf.get("gpumem", "")
    type_name = cls._c_type
    return (
        f"typedef {gpu_memory_qualifier} struct {type_name}_s * {type_name};"
    )


def gen_enum(cls, conf):
    """Generate the C enum listing unionref member types.

    Parameters:
        cls: Unionref class to describe.
        conf: C API generation options. Present for API symmetry.

    Returns:
        C enum declaration statement.
    """
    type_name = cls.__name__
    enum_items = ",".join(
        f"{type_name}_{member._c_type}_t" for member in cls._reftypes
    )
    return f"enum {type_name}_e{{{enum_items}}};"


def methods_from_path(cls, path, conf):
    """Return generated C methods and kernels needed for one access path.

    Parameters:
        cls: Root xobjects class for generated methods.
        path: Access path from root object to target.
        conf: C API generation options.

    Returns:
        List of (C source, Kernel) tuples. Stubs may return (None, None).
    """
    generated_methods = []
    target_type = path[-1]

    if is_scalar(target_type):
        generated_methods.append(gen_method_get(cls, path, conf))
        generated_methods.append(gen_method_set(cls, path, conf))

    if is_type(target_type):
        generated_methods.append(gen_method_getp(cls, path, conf))

    if is_array(target_type):
        generated_methods.append(gen_method_len(cls, path, conf))
        generated_methods.append(gen_method_shape(cls, path, conf))
        generated_methods.append(gen_method_nd(cls, path, conf))
        # Not yet implemented, only method stubs:
        # generated_methods.append(gen_method_strides(cls, path, conf))
        # generated_methods.append(gen_method_getpos(cls, path, conf))

    if is_unionref(target_type):
        generated_methods.append(gen_method_typeid(cls, path, conf))
        generated_methods.append(gen_method_member(cls, path, conf))
        if cls == target_type:
            for method in target_type._methods:
                generated_methods.append(
                    gen_method_switch(cls, path, conf, method)
                )
    return generated_methods


def gen_cdef(cls, conf):
    """Generate C type declarations needed before method declarations.

    Parameters:
        cls: xobjects class to declare.
        conf: C API generation options.

    Returns:
        C type declarations as source text.
    """
    declarations = []
    declarations.append(gen_typedef(cls, conf))
    if is_unionref(cls):
        declarations.append(gen_enum(cls, conf))
    return "\n".join(declarations)


def gen_code(cls, paths, conf):
    """Generate C source for the class C API.

    Parameters:
        cls: xobjects class to generate for.
        paths: Access paths that need generated methods.
        conf: C API generation options.

    Returns:
        Full generated C source text for the class API.
    """
    type_name = cls.__name__
    sources = []
    sources.append(f"#ifndef XOBJ_TYPEDEF_{type_name}")
    sources.append(f"#define XOBJ_TYPEDEF_{type_name}")
    sources.append(gen_cdef(cls, conf))

    generated_methods = []
    for path in paths:
        generated_methods.extend(methods_from_path(cls, path, conf))

    for source, _ in generated_methods:
        if source is not None:
            sources.append(source)

    sources.append(f"#endif")

    source = "\n".join(sources)

    return source


def gen_kernels(cls, paths, conf):
    """Generate Kernel definitions for the class C API.

    Parameters:
        cls: xobjects class to generate for.
        paths: Access paths that need generated kernels.
        conf: C API generation options.

    Returns:
        Dictionary mapping C function names to Kernel descriptions.
    """
    generated_methods = []
    for path in paths:
        generated_methods.extend(methods_from_path(cls, path, conf))

    kernels = {}
    for _, kernel in generated_methods:
        if kernel is not None:
            kernels[kernel.c_name] = kernel

    return kernels


def gen_cdefs(cls, paths, conf):
    """Generate C declarations for the class C API.

    Parameters:
        cls: xobjects class to generate for.
        paths: Access paths that need generated declarations.
        conf: C API generation options.

    Returns:
        C declarations as source text.
    """
    kernels = gen_kernels(cls, paths, conf)

    declarations = [gen_cdef(cls, conf)]

    for _, kernel in kernels.items():
        declarations.append(gen_c_decl_from_kernel(kernel, conf) + ";")

    return "\n".join(declarations)

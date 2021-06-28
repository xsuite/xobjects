from .context import Kernel, Arg

from .scalar import Int64, Void, Int8


def is_field(part):
    return hasattr(part, "ftype")


def is_struct(atype):
    return hasattr(atype, "_fields")


def is_array(atype):
    return hasattr(atype, "_shape")


def is_scalar(atype):
    return hasattr(atype, "_dtype")


def is_ref(atype):
    return hasattr(atype, "_reftype")


def is_unionref(atype):
    return hasattr(atype, "_reftypes")


def is_compound(atype):
    """Types that are referenced in C with an opaque pointer"""
    return is_array(atype) or is_struct(atype) or is_ref(atype)


def get_inner_type(part):
    """type contained in a field, array or ref, else None"""
    if is_field(part):  # is a field
        return part.ftype
    elif is_array(part):  # is an array
        return part._itemtype
    elif is_ref(part):
        return part._type
    elif is_unionref(part):
        return None
    else:
        raise ValueError(f"Cannot get inner type of {part}")


def gen_pointer(ctype, conf):
    gpumem = conf.get("gpumem", "")
    return f"{gpumem}{ctype}"


def gen_c_type_from_arg(arg: Arg, conf):
    gpumem = conf.get("gpumem", "")
    if arg is None:
        cdec = "void"
    else:
        cdec = arg.atype._c_type
        if arg.pointer:
            cdec = f"{gpumem}{cdec}*"
        if arg.const:
            cdec = "const " + cdec
    return cdec


def gen_c_arg_from_arg(arg: Arg, conf):
    gpumem = conf.get("gpumem", "")
    cpurestrict = conf.get("cpurestrict", "")
    if arg is None:
        cdec = "void"
    else:
        cdec = arg.atype._c_type
        if arg.pointer:
            cdec = f"{gpumem}{cdec}*{cpurestrict}"
        elif is_compound(arg.atype):
            cdec = f"{cdec}{cpurestrict}"
        if arg.const:
            cdec = "const " + cdec
    return f"{cdec} {arg.name}"


def gen_c_size_from_arg(arg: Arg, conf):
    if arg is None:
        return None
    else:
        if is_compound(arg.atype):
            return conf.get("pointersize", 8)
        else:
            return arg.atype._size


def gen_c_decl_from_kernel(kernel: Kernel, conf):
    args = ", ".join([gen_c_arg_from_arg(arg, conf) for arg in kernel.args])
    if kernel.ret is None:
        ret = "void"
    else:
        ret = gen_c_type_from_arg(kernel.ret, conf)
    gpufun = conf.get("gpufun")
    if gpufun is None:
        return f"{ret} {kernel.c_name}({args})"
    else:
        return f"{gpufun} {ret} {kernel.c_name}({args})"


def get_layers(parts):
    layers = 0
    for part in parts:
        if hasattr(part, "_shape"):
            layers += 1
    return layers


def int_from_obj(offset, conf):
    inttype = gen_pointer(conf.get("inttype", "int64_t") + "*", conf)
    chartype = gen_pointer(conf.get("chartype", "char") + "*", conf)
    return f"*({inttype})(({chartype}) obj+{offset})"


def Field_get_c_offset(self, conf):
    if self.is_reference:
        doffset = f"offset+{self.offset}"  # starts of data
        refoffset = int_from_obj(doffset, conf)
        return [f"  offset+={refoffset};"]
    else:
        return self.offset


def Ref_get_c_offset(self, conf):
    refoffset = int_from_obj("offset", conf)
    return [f"  offset+={refoffset};"]


def Array_get_c_offset(cls, conf):
    inttype = conf.get("inttype", "int64_t")

    out = []
    if hasattr(cls, "_strides"):  # static shape or 1d dynamic shape
        strides = cls._strides
    else:
        nd = len(cls._shape)
        strides = []
        stride_offset = 8 + nd * 8
        for ii in range(nd):
            sname = f"{inttype} {cls.__name__}_s{ii}"
            svalue = int_from_obj(f"offset+{stride_offset}", conf)
            out.append(f"{sname}={svalue};")
            strides.append(sname)

    soffset = "+".join([f"i{ii}*{ss}" for ii, ss in enumerate(strides)])
    if cls._data_offset > 0:
        soffset = f"{cls._data_offset}+{soffset}"
    if cls._is_static_type:
        out.append(f"  offset+={soffset};")
    else:
        out.append(int_from_obj(f"offset+{soffset}", conf))
    return out


def get_c_offset(atype, conf):
    if is_array(atype):
        return Array_get_c_offset(atype, conf)
    elif is_field(atype):
        return Field_get_c_offset(atype, conf)
    elif is_ref(atype):
        return Ref_get_c_offset(atype, conf)


def gen_method_offset(path, conf):
    """return code to obtain offset of the target in bytes"""
    inttype = conf.get("inttype", "int64_t")
    lst = [f"  {inttype} offset=0;"]
    offset = 0
    for part in path:
        soffset = get_c_offset(part, conf)
        if type(soffset) is int:
            offset += soffset
        else:
            if offset > 0:
                lst.append(f"  offset+={offset};")  # dump current offset
            lst.extend(soffset)  # update reference offset
            offset = 0
    if offset > 0:
        lst.append(f"  offset+={offset};")
    return "\n".join(lst)


def gen_fun_data(cls, parts, action, const, extra, ret):
    typename = cls._c_type
    fields = []
    indices = 0
    for part in parts:
        if is_field(part):  # is field
            fields.append(part.name)
        elif is_array(part):  # is array
            indices += len(part._shape)
    fun_name = [typename, action]
    if len(fields) > 0:
        fun_name.append("_".join(fields))
    fun_name = "_".join(fun_name)
    args = [Arg(cls, pointer=False, const=const, name="obj")]
    for ii in range(indices):
        args.append(Arg(Int64, name=f"i{ii}"))

    args.extend(extra)

    return Kernel(args, c_name=fun_name, ret=ret)


def gen_c_pointed(target: Arg, conf):
    size = gen_c_size_from_arg(target, conf)
    ret = gen_c_type_from_arg(target, conf)
    if target.pointer or is_compound(target.atype):
        chartype = gen_pointer(conf.get("chartype", "char") + "*", conf)
        return f"({ret})(({chartype}) obj+offset)"
    else:
        rettype = gen_pointer(ret + "*", conf)
        if size == 1:
            return f"*(({rettype}) obj+offset)"
        else:
            chartype = gen_pointer(conf.get("chartype", "char") + "*", conf)
            return f"*({rettype})(({chartype}) obj+offset)"


def gen_method_get(cls, path, conf):
    lasttype = get_inner_type(path[-1])
    retarg = Arg(lasttype)
    kernel = gen_fun_data(
        cls,
        path,
        const=True,
        action="get",
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    # lst.append(r'  printf("Obj:  %p\n", (void*) obj);')
    lst.append(gen_method_offset(path, conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append(f"  return {pointed};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_set(cls, path, conf):
    lasttype = get_inner_type(path[-1])
    valarg = Arg(lasttype, name="value")
    kernel = gen_fun_data(
        cls,
        path,
        const=False,
        action="set",
        extra=[valarg],
        ret=None,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    lst.append(gen_method_offset(path, conf))
    pointed = gen_c_pointed(valarg, conf)
    lst.append(f"  {pointed}=value;")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_getp(cls, path, conf):
    lasttype = get_inner_type(path[-1])
    if lasttype is None:
        retarg = Arg(Void, pointer="True")
    else:
        retarg = Arg(lasttype)
    if is_scalar(lasttype):
        retarg.pointer = True

    action = "getp"
    if is_ref(lasttype):
        action += "ref"
    layers = get_layers(path)
    if layers > 0:
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        path,
        const=False,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    lst.append(gen_method_offset(path, conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append(f"  return {pointed};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_len(cls, path, conf):
    lasttype = get_inner_type(path[-1])
    assert is_array(lasttype)

    retarg = Arg(Int64)

    action = "len"
    layers = get_layers(path)
    if layers > 0 and not is_scalar(lasttype):
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        path,
        const=False,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    if lasttype._is_static_shape:
        ll = lasttype._get_n_items()
        lst.append(f"  return {ll};"), kernel
    else:
        lst.append(gen_method_offset(path, conf))
        arrarg = Arg(Int64, pointer=True)
        pointed = gen_c_pointed(arrarg, conf)
        typearr = gen_pointer("int64_t*", conf)
        lst.append(f"{typearr} arr= {pointed};")
        terms = []
        ii = 1
        for sh in lasttype._shape:
            if sh is None:
                terms.append(f"arr[{ii}]")
            else:
                terms.append(str(sh))
        terms = "*".join(terms)
        lst.append(f"  return {terms};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_size(cls, path, conf):
    innertype = get_inner_type(path[-1])
    if innertype is None:  # cannot determine size
        return (None, None)
    retarg = Arg(Int64)

    action = "size"
    layers = get_layers(path)
    if layers > 0 and not is_scalar(innertype):
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        path,
        const=False,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    if innertype._size is None:
        lst.append(gen_method_offset(path, conf))
        pointed = gen_c_pointed(retarg, conf)
        lst.append(f"  return {pointed};")
    else:
        lst.append(f"  return {innertype._size};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_shape(cls, path, conf):
    "return shape in an array"
    return None, None


def gen_method_nd(cls, path, conf):
    "return length of shape"
    return None, None


def gen_method_strides(cls, path, conf):
    "return strides"
    return None, None


def gen_method_getpos(cls, path, conf):
    "return slot position from index and strides"
    return None, None


def gen_method_typeid(cls, path, conf):
    "return typeid of a unionref"
    retarg = Arg(Int64)

    action = "typeid"

    kernel = gen_fun_data(
        cls,
        path,
        const=True,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    lst.append(gen_method_offset(path, conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append("  offset+=8;")
    lst.append(f"  return {pointed};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_member(cls, path, conf):
    "return typeid of a unionref"
    retarg = Arg(Void, pointer=True)

    action = "member"

    kernel = gen_fun_data(
        cls,
        path,
        const=True,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    lst.append(gen_method_offset(path, conf))

    # pointed = gen_c_pointed(Arg(Int8, pointer=True), conf)
    # GPU not handled
    lst.append("  char *reloff_p = (char *) obj + offset;")
    lst.append("  int64_t  reloff= *(int64_t *) reloff_p;")
    lst.append("  return (void *) (reloff_p + reloff);")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_typedef(cls, conf):
    gpumem = conf.get("gpumem", "")
    typename = cls._c_type
    return f"typedef {gpumem} struct {typename}_s * {typename};"


def gen_enum(cls, conf):
    st = ",".join(f"{tt._c_type}_t" for tt in cls._reftypes)
    return f"enum {typename}_e{{{lst}}};"


def methods_from_path(cls, path, conf):
    """
    size: all
    get,set: innertype is scalar
    getp: all but union ref as innertype or lasttype

    """
    out = []
    lasttype = path[-1]
    innertype = get_inner_type(lasttype)

    if is_scalar(innertype):
        out.append(gen_method_get(cls, path, conf))
        out.append(gen_method_set(cls, path, conf))
    else:
        out.append(gen_method_size(cls, path, conf))

    if is_array(innertype):
        out.append(gen_method_len(cls, path, conf))
        out.append(gen_method_shape(cls, path, conf))
        out.append(gen_method_nd(cls, path, conf))
        out.append(gen_method_strides(cls, path, conf))
        out.append(gen_method_getpos(cls, path, conf))

    if is_unionref(innertype):
        out.append(gen_method_typeid(cls, path, conf))
        out.append(gen_method_member(cls, path, conf))

    if not (is_unionref(lasttype) or is_unionref(lasttype)):
        out.append(gen_method_getp(cls, path, conf))

    return out


def gen_cdef(cls, conf):
    out = []
    out.append(gen_typedef(cls, conf))
    if is_unionref(cls):
        out.append(gen_enum(cls, conf))
    return "\n".join(out)


def gen_code(cls, paths, conf):
    """
    Generate source code, used by add_kernels

    Provide all symbols starting from cls.__name__

    The definition must be completed by inner classes if present
    """

    typename = cls.__name__
    sources = []
    sources.append(f"#ifndef XOBJ_TYPEDEF_{typename}")
    sources.append(f"#define XOBJ_TYPEDEF_{typename}")
    sources.append(gen_cdef(cls, conf))

    out = []
    for path in paths:
        out.extend(methods_from_path(cls, path, conf))

    for source, kernel in out:
        if source is not None:
            sources.append(source)

    sources.append(f"#endif")

    source = "\n".join(sources)

    return source


def gen_kernels(cls, paths, conf):
    """
    Generate kernel defintions of the C Api

    """
    out = []
    for path in paths:
        out.extend(methods_from_path(cls, path, conf))

    kernels = []
    for source, kernel in out:
        if kernel is not None:
            kernels.append(kernel)

    return kernels


def gen_cdefs(cls, paths, conf):
    """

    Generate kernel defintions of the C Api

    """

    kernels = gen_kernels(cls, paths, conf)

    out = [gen_cdef(cls, conf)]

    for kernel in kernels:
        out.append(gen_c_decl_from_kernel(kernel, conf) + ";")

    return "\n".join(out)

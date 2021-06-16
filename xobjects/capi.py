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
    return hasattr(atype, "_rtypes")


def is_unionref(atype):
    return hasattr(atype, "_reftypes")


def is_single_ref(atype):
    return hasattr(atype, "_rtypes") and len(atype._rtypes) == 1


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
        if len(part._rtypes) > 1:
            return None
        else:
            return part._rtypes[0]
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
    return f"{ret} {kernel.c_name}({args})"


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
        # soffset = part._get_c_offset(conf)
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


def gen_method_get(path, conf):
    cls = path[0]
    path = path[1:]
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


def gen_method_set(path, conf):
    cls = path[0]
    path = path[1:]
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


def gen_method_getp(path, conf):
    cls = path[0]
    path = path[1:]
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


def gen_method_len(path, conf):
    cls = path[0]
    path = path[1:]
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


def gen_method_size(path, conf):
    cls = path[0]
    path = path[1:]
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


def gen_method_shape(path, conf):
    "return shape in an array"
    return None, None


def gen_method_nd(path, conf):
    "return length of shape"
    return None, None


def gen_method_strides(path, conf):
    "return strides"
    return None, None


def gen_method_getpos(path, conf):
    "return slot position from index and strides"
    return None, None


def gen_method_typeid(path, conf):
    "return typeid of a unionref"
    cls = path[0]
    path = path[1:]
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


def gen_method_member(path, conf):
    "return typeid of a unionref"
    cls = path[0]
    path = path[1:]
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


def gen_typedef_decl(cls, conf):
    out = []
    typename = cls._c_type
    if is_struct(cls) or is_array(cls) or is_single_ref(cls):
        out.append(f"#ifndef XOBJ_TYPEDEF_{typename}")
        out.append(f"{gen_typedef(cls,conf)}")
        out.append(f"#define XOBJ_TYPEDEF_{typename}")
        out.append("#endif")
    elif is_unionref(cls):
        out.append(f"#ifndef XOBJ_TYPEDEF_{typename}")
        # defining C union might have issues with GPU qualifiers
        out.append(f"{gen_typedef(cls,conf)}")
        lst = ",".join(f"{tt._c_type}_t" for tt in cls._reftypes)
        out.append(f"enum {typename}_e{{{lst}}};")
        out.append(f"#define XOBJ_TYPEDEF_{typename}")
        out.append("#endif")
    return "\n".join(out)


def gen_headers(paths, conf):
    out = []

    types = {}
    for path in paths:
        types[path[0]._c_type] = path[0]
        for part in path[1:]:
            lasttype = get_inner_type(part)
            if lasttype is not None:
                types[lasttype._c_type] = lasttype
    for _, tt in types.items():
        out.append(gen_typedef_decl(tt, conf))
    return "\n".join(out)


def gen_cdef(paths, conf):
    types = {}
    out = []
    for path in paths:
        types[path[0]._c_type] = path[0]
        for part in path[1:]:
            lasttype = get_inner_type(part)
            if lasttype is not None:
                types[lasttype._c_type] = lasttype
    for _, tt in types.items():
        if not is_scalar(tt):
            out.append(gen_typedef(tt, conf))
    return "\n".join(out)


def methods_from_path(path, conf):
    """
    size: all
    get,set: innertype is scalar
    getp: all but union ref as innertype or lasttype



    """
    out = []
    lasttype = path[-1]
    innertype = get_inner_type(lasttype)

    if is_scalar(innertype):
        out.append(gen_method_get(path, conf))
        out.append(gen_method_set(path, conf))
    else:
        out.append(gen_method_size(path, conf))

    if is_array(innertype):
        out.append(gen_method_len(path, conf))
        out.append(gen_method_shape(path, conf))
        out.append(gen_method_nd(path, conf))
        out.append(gen_method_strides(path, conf))
        out.append(gen_method_getpos(path, conf))

    if is_unionref(innertype):
        out.append(gen_method_typeid(path, conf))
        out.append(gen_method_member(path, conf))

    if not (is_unionref(lasttype) or is_unionref(lasttype)):
        out.append(gen_method_getp(path, conf))

    return out


def gen_code(paths, conf):
    out = []
    for path in paths:
        out.extend(methods_from_path(path, conf))

    sources = [gen_headers(paths, conf)]
    kernels = {}
    for source, kernel in out:
        if source is not None:
            sources.append(source)
        if kernel is not None:
            kernels[kernel.c_name] = kernel

    source = "\n".join(sources)

    cdef = gen_cdef(paths, conf)
    return source, kernels, cdef

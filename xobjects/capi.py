# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from .context import Kernel, Arg

from .scalar import Int64, Void, Int8, is_scalar
from .struct import is_field, is_struct
from .array import is_index, is_array
from .ref import is_unionref, is_ref
from .string import is_string


def is_compound(atype):
    """Types that are referenced in C with an opaque pointer"""
    return is_struct(atype) or is_array(atype) or is_unionref(atype)


def is_type(atype):
    return is_compound(atype) or is_scalar(atype) or is_string(atype)


def get_inner_type(part):
    """type contained in a field, array or ref, else None"""
    if is_field(part):  # is a field
        return part.ftype
    elif is_array(part):  # is an array
        return part._itemtype
    elif is_ref(part):
        return part._reftype
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


def Index_get_c_offset(part, conf, icount):
    cls = part.cls
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

    soffset = "+".join([f"i{ii+icount}*{ss}" for ii, ss in enumerate(strides)])
    if cls._data_offset > 0:
        soffset = f"{cls._data_offset}+{soffset}"
    if cls._is_static_type:
        out.append(f"  offset+={soffset};")
    else:
        out.append(int_from_obj(f"offset+{soffset}", conf))
    return out


def gen_method_offset(path, conf):
    """return code to obtain offset of the target in bytes"""
    inttype = conf.get("inttype", "int64_t")
    lst = [f"  {inttype} offset=0;"]
    offset = 0
    icount = 0
    for part in path:
        if is_index(part):
            soffset = Index_get_c_offset(part, conf, icount)
            icount += len(part.cls._shape)
        elif is_field(part):
            soffset = Field_get_c_offset(part, conf)
        elif is_ref(part):
            soffset = Ref_get_c_offset(part, conf)
        else:
            soffset = None
        if type(soffset) is int:
            offset += soffset
        elif type(soffset) is list:
            if offset > 0:
                lst.append(f"  offset+={offset};")  # dump current offset
            lst.extend(soffset)  # update reference offset
            offset = 0
    if offset > 0:
        lst.append(f"  offset+={offset};")
    return "\n".join(lst)


def gen_fun_kernel(cls, path, action, const, extra, ret, add_nindex=False):
    typename = cls._c_type
    fields = []
    indices = 0
    for part in path:
        if is_field(part):  # is field
            fields.append(part.name)
        elif is_index(part):  # is array
            indices += len(part.cls._shape)
    if add_nindex and indices > 0:
        action += str(indices)
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
    lasttype = path[-1]
    retarg = Arg(lasttype)
    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="get",
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


def gen_method_set(cls, path, conf):
    lasttype = path[-1]
    valarg = Arg(lasttype, name="value")
    kernel = gen_fun_kernel(
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
    lasttype = path[-1]
    if lasttype is None:
        retarg = Arg(Void, pointer="True")
    else:
        retarg = Arg(lasttype)
    if is_scalar(lasttype):
        retarg.pointer = True

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="getp",
        extra=[],
        ret=retarg,
        add_nindex=True,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    lst.append(gen_method_offset(path, conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append(f"  return {pointed};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_len(cls, path, conf):
    lasttype = path[-1]

    retarg = Arg(Int64)

    kernel = gen_fun_kernel(
        cls,
        path,
        const=False,
        action="len",
        extra=[],
        ret=retarg,
        add_nindex=True,
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
        lst.append(f"  {typearr} arr= {pointed};")
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

    kernel = gen_fun_kernel(
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

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="typeid",
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

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action="member",
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    lst.append(gen_method_offset(path, conf))

    # pointed = gen_c_pointed(Arg(Int8, pointer=True), conf)
    lst.extend(Ref_get_c_offset("offset", conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append(f" return {pointed};")
    # GPU not handled
    # lst.append("  char *reloff_p = (char *) obj + offset;")
    # lst.append("  int64_t  reloff= *(int64_t *) reloff_p;")
    # lst.append("  return (void*) (reloff_p + reloff);")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_switch(cls, path, conf, method):
    "generate switch methods declared in _methods"
    lasttype = path[-1]

    kernel = gen_fun_kernel(
        cls,
        path,
        const=True,
        action=method.c_name,
        extra=method.args,
        ret=method.ret,
    )
    refname = lasttype.__name__

    decl = gen_c_decl_from_kernel(kernel, conf)
    lst = [decl + "{"]
    voidp = gen_pointer("void*", conf)
    lst.append(f"  {voidp} member = {refname}_member(obj);")
    lst.append(f"  switch ({refname}_typeid(obj)){{")
    for atype in lasttype._reftypes:
        atname = atype.__name__
        targs = [f"({atname}) member"]
        for arg in kernel.args[1:]:
            targs.append(f"{arg.name}")
        targs = ",".join(targs)
        lst.append(
            f"""\
        #ifndef {refname.upper()}_SKIP_{atname.upper()}
        case {refname}_{atname}_t:
            return {atname}_{method.c_name}({targs});
            break;
        #endif"""
        )
    lst.append("  }")
    lst.append("  return 0;")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_typedef(cls, conf):
    gpumem = conf.get("gpumem", "")
    typename = cls._c_type
    return f"typedef {gpumem} struct {typename}_s * {typename};"


def gen_enum(cls, conf):
    typename = cls.__name__
    lst = ",".join(f"{typename}_{tt._c_type}_t" for tt in cls._reftypes)
    return f"enum {typename}_e{{{lst}}};"


def methods_from_path(cls, path, conf):
    """
    size: all
    get, set: lasttype is scalar
    getp: all real types

    """
    out = []
    lasttype = path[-1]

    if is_scalar(lasttype):
        out.append(gen_method_get(cls, path, conf))
        out.append(gen_method_set(cls, path, conf))

    if is_type(lasttype):
        out.append(gen_method_getp(cls, path, conf))

    if is_array(lasttype):
        out.append(gen_method_len(cls, path, conf))
    #    out.append(gen_method_shape(cls, path, conf))
    #    out.append(gen_method_nd(cls, path, conf))
    #    out.append(gen_method_strides(cls, path, conf))
    #    out.append(gen_method_getpos(cls, path, conf))

    if is_unionref(lasttype):
        out.append(gen_method_typeid(cls, path, conf))
        out.append(gen_method_member(cls, path, conf))
        if cls == lasttype:
            for method in lasttype._methods:
                out.append(gen_method_switch(cls, path, conf, method))
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

    for source, _ in out:
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

    kernels = {}
    for _, kernel in out:
        if kernel is not None:
            kernels[kernel.c_name] = kernel

    return kernels


def gen_cdefs(cls, paths, conf):
    """

    Generate kernel defintions of the C Api

    """

    kernels = gen_kernels(cls, paths, conf)

    out = [gen_cdef(cls, conf)]

    for _, kernel in kernels.items():
        out.append(gen_c_decl_from_kernel(kernel, conf) + ";")

    return "\n".join(out)

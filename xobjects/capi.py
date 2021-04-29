from .context import Kernel, Arg

from .scalar import Int64


def get_last_type(specs, conf):
    prepointer = conf.get("prepointer", "")
    pointersize = conf.get("pointersize", 64)
    spec = specs[-1]
    if hasattr(spec, "name"):  # is a field
        lasttype = spec.ftype
    elif hasattr(spec, "_shape"):  # is an array
        lasttype = spec._itemtype

    if hasattr(lasttype, "_c_type"):
        ret = lasttype._c_type
        size = lasttype._size
    else:
        ret = f"{lasttype.__name__}*"
        size = pointersize

    if "*" in ret and prepointer != "":
        ret = f"{prepointer} {ret}"

    return ret, size


def get_last_scalar_type(specs):
    spec = specs[-1]
    if hasattr(spec, "_dtype"):
        return spec._c_type
    else:
        return None


def get_last_type2(specs):
    spec = specs[-1]
    if hasattr(spec, "name"):  # is a field
        lasttype = spec.ftype
    elif hasattr(spec, "_shape"):  # is an array
        lasttype = spec._itemtype
    if hasattr(lasttype, "_c_type"):
        ret = ("scalar", lasttype._c_type)
    else:
        ret = ("pointer", lasttype._c_type)
    return ret


# new take


def is_field(part):
    return hasattr(part, "ftype")


def is_struct(atype):
    return hasattr(atype, "_fields")


def is_xobject(atype):
    return is_array(atype) or is_struct(atype)


def is_array(atype):
    return hasattr(atype, "_shape")


def is_scalar(atype):
    return hasattr(atype, "_dtype")


def get_inner_type(part):
    if is_field(part):  # is a field
        return part.ftype
    elif is_array(part):  # is an array
        return part._itemtype


def get_inner_c_type(part):
    if is_field(part):  # is a field
        return part.ftype._c_type
    elif is_array(part):  # is an array
        return part._itemtype._c_type


def is_last_scalar(spec):
    atype = get_inner_type(spec[-1])
    if atype is not None:
        return is_scalar(atype)
    else:
        return False


def is_last_array(spec):
    atype = get_inner_type(spec[-1])
    if atype is not None:
        return hasattr(atype, "_shape")
    else:
        return False


def gen_pointer(typename, argname, conf):
    prepointer = conf.get("prepointer", "")
    postpointer = conf.get("postpointer", "")
    return f"{prepointer}{typename}*{postpointer} {argname}"


def gen_arg(atype, conf, argname="", const=False, pointer=False):
    ctype = atype._c_type
    if pointer:
        ctype += "*"
    if const:
        ctype = "const " + ctype
    if pointer or is_xobject(atype):
        ctype = dress_pointer(ctype, conf)
    if argname != "":
        ctype = f"{ctype} {argname}"
    return ctype


def dress_pointer(ctype, conf):
    prepointer = conf.get("prepointer", "")
    postpointer = conf.get("postpointer", "")
    return f"{prepointer}{ctype}{postpointer}"


def gen_int(argname, conf):
    itype = conf.get("itype", "int64_t")
    return f"{itype} {argname}"


def gen_return(ret, pointer, conf):
    if hasattr(ret, "_c_type"):
        ctype = ret._c_type
    else:
        ctype = ret
    if pointer or is_xobject(ret.atype):
        ctype = dress_pointer(ctype, conf)
    return ctype


def gen_method_offset(specs, conf):
    """return code to obtain offset of the target in bytes"""
    itype = conf.get("itype", "int64_t")
    lst = [f"  {itype} offset=0;"]
    offset = 0
    for spec in specs:
        soffset = spec._get_c_offset(conf)
        if type(soffset) is int:
            offset += soffset
        else:
            lst.append(f"  offset+={offset};")  # dump current offset
            lst.extend(soffset)  # update reference offset
            offset = 0
    if offset > 0:
        lst.append(f"  offset+={offset};")
    return "\n".join(lst)


def c_type_from_arg(arg: Arg, conf):
    if arg is None:
        return "void"
    else:
        cdec = arg.atype._c_type
        if arg.pointer:
            cdec = dress_pointer(cdec + "*", conf)
        if is_xobject(arg.atype):
            cdec = dress_pointer(cdec, conf)
        if arg.const:
            cdec = "const " + cdec
        return cdec


def c_arg_from_arg(arg: Arg, conf):
    cdec = c_type_from_arg(arg, conf)
    return f"{cdec} {arg.name}"


def c_size_from_arg(arg: Arg, conf):
    if arg is None:
        return None
    else:
        if arg.pointer:
            return conf.get("pointersize", 64)
        elif is_xobject(arg.atype):
            return conf.get("pointersize", 64)
        else:
            return arg.atype._size


def gen_c_decl_from_kernel(kernel: Kernel, conf):
    args = ", ".join([c_arg_from_arg(arg, conf) for arg in kernel.args])
    if kernel.ret is None:
        ret = "void"
    else:
        ret = c_type_from_arg(kernel.ret, conf)
    return f"{ret} {kernel.c_name}({args})"


def gen_fun_data(cls, parts, action, const, extra, ret):
    typename = cls._c_type
    nparts = []
    indices = 0
    for part in parts:
        if hasattr(part, "name"):  # is field
            nparts.append(part.name)
        elif hasattr(part, "_shape"):  # is array
            indices += len(part._shape)
    npart = "_".join(nparts)
    fun_name = f"{typename}_{action}_{npart}"
    args = [Arg(cls, pointer=False, const=const, name="obj")]
    for ii in range(indices):
        args.append(Arg(Int64, name=f"i{ii}"))

    args.extend(extra)

    return Kernel(args, c_name=fun_name, ret=ret)


def gen_c_pointed(target: Arg, conf):
    size = c_size_from_arg(target, conf)
    ret = c_type_from_arg(target, conf)
    if target.pointer or is_xobject(target.atype):
        return f"({ret})((char*) obj+offset)"
    else:
        if size == 1:
            return f"*(({ret}*) obj+offset)"
        else:
            return f"*({ret}*)((char*) obj+offset)"


def gen_method_get(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    retarg = Arg(lasttype)
    kernel = gen_fun_data(
        cls,
        parts,
        const=True,
        action="get",
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    lst.append(gen_method_offset(parts, conf))
    pointed = gen_c_pointed(retarg, conf)
    lst.append(f"  return {pointed};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_set(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    valarg = Arg(lasttype, name="value")
    kernel = gen_fun_data(
        cls,
        parts,
        const=False,
        action="set",
        extra=[valarg],
        ret=None,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]
    lst.append(gen_method_offset(parts, conf))
    pointed = gen_c_pointed(valarg, conf)
    lst.append(f"  {pointed}=value;")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_getp(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    kernel = gen_fun_data(
        cls,
        parts,
        const=False,
        action="getp",
        extra=[],
        ret=Arg(lasttype),
    )
    decl = gen_c_decl_from_kernel(kernel, conf)
    return None, None
    return decl + ";", kernel


def gen_len(cls, parts, conf):
    return None, None


def gen_size(cls, parts, conf):
    return None, None


def gen_dim(cls, parts, conf):
    return None, None


def gen_ndim(cls, parts, conf):
    return None, None


def gen_strides(cls, parts, conf):
    return None, None


def gen_iter(cls, parts, conf):
    return None, None


def gen_typedef(cls):
    typename = cls._c_type
    return f"typedef struct {typename} * {typename};"


def gen_typedef_decl(cls):
    # TODO: for Union add enums
    out = []
    typename = cls._c_type
    if not is_scalar(cls):
        out.append(
            f"""
#ifndef XOBJ_TYPEDEF_{typename}
{gen_typedef(cls)}
#define XOBJ_TYPEDEF_{typename}
#endif
"""
        )
    return "\n".join(out)


def gen_headers(cls, specs):
    out = ["#include <stdint.h>"]
    types = set()
    types.add(cls)
    for parts in specs:
        for part in parts:
            types.add(get_inner_type(part))
    for tt in types:
        out.append(gen_typedef_decl(tt))
    return "\n".join(out)


def gen_cdef(cls, specs):
    types = set()
    types.add(cls)
    out = []
    for parts in specs:
        for part in parts:
            types.add(get_inner_type(part))
    for tt in types:
        if not is_scalar(tt):
            out.append(gen_typedef(tt))
    return "\n".join(out)


def gen_code(cls, specs, conf):
    out = []
    for parts in specs:
        out.append(gen_getp(cls, parts, conf))
        out.append(gen_size(cls, parts, conf))
        if is_last_scalar(parts):
            out.append(gen_method_get(cls, parts, conf))
            out.append(gen_method_set(cls, parts, conf))
        if is_last_array(parts):
            out.append(gen_len(cls, parts, conf))
            out.append(gen_dim(cls, parts, conf))
            out.append(gen_ndim(cls, parts, conf))
            out.append(gen_strides(cls, parts, conf))
            out.append(gen_iter(cls, parts, conf))

    sources = [gen_headers(cls, specs)]
    kernels = {}
    for source, kernel in out:
        if source is not None:
            sources.append(source)
        if kernel is not None:
            kernels[kernel.c_name] = kernel

    source = "\n".join(sources)

    cdef = gen_cdef(cls, specs)
    return source, kernels, cdef

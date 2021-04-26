from typing import NamedTuple, List


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


def gen_method_get_declaration(name, specs, conf):
    itype = conf.get("itype", "int64_t")
    prepointer = conf.get("prepointer", "")
    postpointer = conf.get("postpointer", "")
    nparts = []
    iparts = 0
    for spec in specs:
        if hasattr(spec, "name"):
            nparts.append(spec.name)
        elif hasattr(spec, "_shape"):
            iparts += len(spec._shape)

    ret = get_last_scalar_type(specs)
    if ret is not None:
        method = f"{ret} {name}_get"
        if len(nparts) > 0:
            method += "_" + "_".join(nparts)
        args = [f"{prepointer}{name}*{postpointer} obj"]
        if iparts > 0:
            args.extend([f"{itype} i{ii}" for ii in range(iparts)])
        return f"{method}({', '.join(args)})"


def gen_method_get_description(name, specs, conf):
    nparts = []
    iparts = 0
    for spec in specs:
        if hasattr(spec, "name"):
            nparts.append(spec.name)
        elif hasattr(spec, "_shape"):
            iparts += len(spec._shape)

    ret = get_last_type2(specs)

    fun_name = f"{name}_get"
    if len(nparts) > 0:
        fun_name += "_" + "_".join(nparts)
    args = [(("pointer", name), "obj")]
    if iparts > 0:
        args.extend([(("scalar", name), f"i{ii}") for ii in range(iparts)])
    return (fun_name, {"args": args, "return": ret})


def gen_method_offset(specs, conf):
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


def gen_method_get_definition(name, specs, conf):
    prepointer = conf.get("prepointer", "")

    lst = [gen_method_get_declaration(name, specs, conf) + "{"]
    lst.append(gen_method_offset(specs, conf))
    ret, size = get_last_type(specs, conf)
    if prepointer != "":
        ret = prepointer + " " + ret
    if "*" in ret:  # return type is a pointer
        lst.append(f"  return ({ret})((char*) obj+offset);")
    else:  # return type is a scalar
        if size == 1:
            lst.append(f"  return *(({ret}*) obj+offset);")
        else:
            lst.append(f"  return *({ret}*)((char*) obj+offset);")
    lst.append("}")
    return "\n".join(lst)


### new take


def is_field(part):
    return hasattr(part, "name")


def is_struct(atype):
    return hasattr(atype, "name")


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
    if pointer or not is_scalar(atype):
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
    if pointer:
        ctype = dress_pointer(ctype, conf)
    return ctype


def gen_c_decl(cls, action, parts, extra, ret, conf):
    typename = cls._c_type
    nparts = []
    iparts = 0
    for part in parts:
        if hasattr(part, "name"):  # is field
            nparts.append(part.name)
        elif hasattr(part, "_shape"):  # is array
            iparts += len(part._shape)
    npart = "_".join(nparts)
    iparts = [gen_int(f"i{ii}", conf) for ii in range(iparts)]
    fname = f"{typename}_{action}_{npart}"
    const = True
    if action in ["set"]:
        const = False
    args = [gen_arg(cls, conf, argname="obj", const=const)]
    args.extend(iparts)
    if extra is not None:
        args.extend(extra)
    args = ", ".join(args)
    return f"{ret} {fname}({args})"


def gen_get(cls, parts, header, conf):
    lasttype = get_inner_type(parts[-1])
    ret = gen_arg(lasttype, conf)
    extra = []
    decl = gen_c_decl(cls, "get", parts, extra, ret, conf)
    if header:
        return decl + ";"
    else:
        prepointer = conf.get("prepointer", "")
        lst = [decl + "{"]
        lst.append(gen_method_offset(parts, conf))
        ret, size = get_last_type(parts, conf)
        if prepointer != "":
            ret = prepointer + " " + ret
        if "*" in ret:  # return type is a pointer
            lst.append(f"  return ({ret})((char*) obj+offset);")
        else:  # return type is a scalar
            if size == 1:
                lst.append(f"  return *(({ret}*) obj+offset);")
            else:
                lst.append(f"  return *({ret}*)((char*) obj+offset);")
        lst.append("}")
        return "\n".join(lst)


def gen_set(cls, parts, header, conf):
    lasttype = get_inner_type(parts[-1])
    ret = "void"
    extra = [gen_arg(lasttype, conf, argname="value")]
    decl = gen_c_decl(cls, "set", parts, extra, ret, conf)
    if header:
        return decl + ";"
    else:
        pass


def gen_getp(cls, parts, header, conf):
    lasttype = get_inner_type(parts[-1])
    ret = gen_arg(lasttype, conf, pointer=True)
    extra = []
    decl = gen_c_decl(cls, "getp", parts, extra, ret, conf)
    if header:
        return decl + ";"
    else:
        pass


def gen_len(cls, parts, header, conf):
    pass


def gen_size(cls, parts, header, conf):
    pass


def gen_dim(cls, parts, header, conf):
    pass


def gen_ndim(cls, parts, header, conf):
    pass


def gen_strides(cls, parts, header, conf):
    pass


def gen_iter(cls, parts, header, conf):
    pass


def gen_functions(cls, specs, header, conf):
    out = []
    for parts in specs:
        out.append(gen_getp(cls, parts, header, conf))
        out.append(gen_size(cls, parts, header, conf))
        if is_last_scalar(parts):
            out.append(gen_get(cls, parts, header, conf))
            out.append(gen_set(cls, parts, header, conf))
        if is_last_array(parts):
            out.append(gen_len(cls, parts, header, conf))
            out.append(gen_dim(cls, parts, header, conf))
            out.append(gen_ndim(cls, parts, header, conf))
            out.append(gen_strides(cls, parts, header, conf))
            out.append(gen_iter(cls, parts, header, conf))
    return out


def gen_typedef(cls):
    # out=["#include <stdint.h>"]
    out = []
    typename = cls._c_type
    if not is_scalar(cls):
        out.append(f"typedef struct {typename} * {typename};")
    return "\n".join(out)


def gen_headers(cls, specs, conf):
    out = []
    out.append(gen_typedef(cls))
    return gen_functions(cls, specs, False, conf)


def gen_code(specs, conf):
    pass

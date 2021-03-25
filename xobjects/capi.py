from typing import NamedTuple, List


def get_last_type(specs, conf):
    prepointer = conf.get("prepointer", "")
    pointersize = conf.get("pointersize", 64)
    spec = specs[-1]
    if hasattr(spec, "name"):  # is a field
        lasttype = spec.ftype
    elif hasattr(spec, "_shape"):  # is an array
        lasttype = spec._itemtype

    if hasattr(lasttype, "_cname"):
        ret = lasttype._cname
        size = lasttype._size
    else:
        ret = f"{lasttype.__name__}*"
        size = pointersize

    if "*" in ret and prepointer != "":
        ret = f"{prepointer} {ret}"

    return ret, size


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

    ret, _ = get_last_type(specs, conf)

    method = f"{ret} {name}_get"
    if len(nparts) > 0:
        method += "_" + "_".join(nparts)
    args = [f"{prepointer}{name}*{postpointer} obj"]
    if iparts > 0:
        args.extend([f"{itype} i{ii}" for ii in range(iparts)])
    return f"{method}({', '.join(args)})"


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
            lst.append(f"  offset+={soffset};")  # update reference offset
            offset = 0
    if offset > 0:
        lst.append(f"  offset+={offset};")
    return "\n".join(lst)


def gen_method_get_body(name, specs, conf):
    itype = conf.get("itype", "int64_t")
    prepointer = conf.get("prepointer", "")
    postpointer = conf.get("postpointer", "")
    ispointer32 = conf.get("ispointer", False)

    lst = [gen_method_get_declaration(name, specs, conf) + "{"]
    lst.append(gen_method_offset(specs, conf))
    ret, size = get_last_type(specs, conf)
    if "*" in ret:  # return type is a pointer
        lst.append(f"  return ({ret})(((char*) obj)+offset);")
    else:  # return type is a scalar
        if size == 1:
            lst.append(f"  return (({ret}*) obj)[offset];")
        else:
            lst.append(f"  return *(({ret} *)(((char*) obj)+offset));")
    lst.append("}")
    return "\n".join(lst)

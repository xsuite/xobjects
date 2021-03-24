from typing import NamedTuple, List


def get_position(specs):
    offset = 0
    offlist = []
    for spec in specs:
        if spec.is_deferred:
            offlist.append(
                f"  offset+=(int64_t *) obj[offset+spec.get_offset()];"
            )
        else:
            offlist.append(f"  offset+={spec.get_offset()};")
    return "\n".join(offlist)


def gen_method_get_signature(
    name, parts, itype="int64_t", prepointer="", postpointer=""
):
    nparts = []
    iparts = 0
    for spec in parts:
        if hasattr(spec, "name"):
            nparts.append(spec.name)
            lasttype = spec.ftype
        elif hasattr(spec, "_shape"):
            iparts += len(spec._shape)
            lasttype = spec._itemtype

    if hasattr(lasttype, "_cname"):
        ret = lasttype._cname
    else:
        ret = f"{lasttype.__name__}*"

    if "*" in ret and prepointer != "":
        ret = f"{prepointer} {ret}"

    method = f"{ret} {name}_get"
    if len(nparts) > 0:
        method += "_" + "_".join(nparts)
    args = [f"{prepointer}{name}*{postpointer} obj"]
    if iparts > 0:
        args.extend([f"{itype} i{ii}" for ii in range(iparts)])
    return f"{method}({', '.join(args)})"

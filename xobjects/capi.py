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
    name, parts, itype="int64", prepointer="", postpointer=""
):
    nparts = []
    iparts = 0
    for spec in parts:
        if hasattr(spec, "name"):
            nparts.append(spec.name)
        elif hasattr(spec, "shape"):
            iparts += len(spec.shape)
    method = "{name}_get"
    if len(nparts) > 0:
        method += "_".join(nparts)
    args = [f"{prepointer}{name}*{postpointer} obj"]
    if inames > 0:
        args.extend([f"{itype} i{ii}" for ii in range(inames)])
    return f"{name}_{fnames}({','.join(inames)})"

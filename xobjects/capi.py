from .context import Kernel, Arg

from .scalar import Int64


def is_field(part):
    return hasattr(part, "ftype")


def is_struct(atype):
    return hasattr(atype, "_fields")


def is_compound(atype):
    """Types that are referenced in C with an opaque pointer"""
    return is_array(atype) or is_struct(atype)


def is_array(atype):
    return hasattr(atype, "_shape")


def is_scalar(atype):
    return hasattr(atype, "_dtype")


def is_ref(atype):
    return hasattr(atype, "_rtypes")


def is_single_ref(atype):
    return hasattr(atype, "_rtypes") and len(atype._rtypes) == 1


def get_inner_type(part):
    """type contained in a field or array"""
    if is_field(part):  # is a field
        return part.ftype
    elif is_array(part):  # is an array
        return part._itemtype
    elif is_ref(part):
        if len(part._rtypes) > 1:
            return part
        else:
            return part._rtypes[0]
    else:
        raise ValueError(f"Cannot get inner type of {part}")


def dress_pointer(ctype, conf):
    prepointer = conf.get("prepointer", "")
    postpointer = conf.get("postpointer", "")
    return f"{prepointer}{ctype}{postpointer}"


def gen_c_type_from_arg(arg: Arg, conf):
    if arg is None:
        return "void"
    else:
        cdec = arg.atype._c_type
        if arg.pointer:
            cdec = dress_pointer(cdec + "*", conf)
        if is_compound(arg.atype):
            cdec = dress_pointer(cdec, conf)
        if arg.const:
            cdec = "const " + cdec
        return cdec


def gen_c_arg_from_arg(arg: Arg, conf):
    cdec = gen_c_type_from_arg(arg, conf)
    return f"{cdec} {arg.name}"


def gen_c_size_from_arg(arg: Arg, conf):
    if arg is None:
        return None
    else:
        if arg.pointer:
            return conf.get("pointersize", 64)
        elif is_compound(arg.atype):
            return conf.get("pointersize", 64)
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
    itype = conf.get("itype", "int64_t")
    ctype = conf.get("ctype", "char")
    return f"({itype})(({ctype}*) obj+{doffset})"


def Field_get_c_offset(self, conf):
    itype = conf.get("itype", "int64_t")
    ctype = conf.get("ctype", "char")
    if self.is_reference:
        doffset = f"offset+{self.offset}"  # starts of data
    else:
        return [f"  offset+={int_from_obj(doffest,)};"]  # WRONG
        return self.offset


def Array_get_c_offset(cls, conf):
    ctype = conf.get("ctype", "char")
    itype = conf.get("itype", "int64_t")

    out = []
    if hasattr(cls, "_strides"):  # static shape or 1d dynamic shape
        strides = cls._strides
    else:
        nd = len(cls._shape)
        strides = []
        stride_offset = 8 + nd * 8
        for ii in range(nd):
            sname = f"{itype} {cls.__name__}_s{ii}"
            svalue = f"*({itype}*) (({ctype}*) obj+offset+{stride_offset})"
            out.append(f"{sname}={svalue};")
            strides.append(sname)

    soffset = "+".join([f"i{ii}*{ss}" for ii, ss in enumerate(strides)])
    if cls._data_offset > 0:
        soffset = f"{cls._data_offset}+{soffset}"
    if cls._is_static_type:
        out.append(f"  offset+={soffset};")
    else:
        out.append(f"  offset+=*({itype}*) (({ctype}*) obj+offset+{soffset});")
    return out


def get_c_offset(atype, conf):
    if is_array(atype):
        return Array_get_c_offset(atype, conf)
    elif is_field(atype):
        return Field_get_c_offset(atype, conf)


def gen_method_offset(path, conf):
    """return code to obtain offset of the target in bytes"""
    itype = conf.get("itype", "int64_t")
    lst = [f"  {itype} offset=0;"]
    offset = 0
    for part in path:
        soffset = part._get_c_offset(conf)
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
    size = gen_c_size_from_arg(target, conf)
    ret = gen_c_type_from_arg(target, conf)
    if target.pointer or is_compound(target.atype):
        return f"({ret})((/*gpuglmem*/ char*) obj+offset)"
    else:
        if size == 1:
            return f"*((/*gpuglmem*/ {ret}*) obj+offset)"
        else:
            return f"*(/*gpuglmem*/{ret}*)((/*gpuglmem*/char*) obj+offset)"


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
    # lst.append(r'  printf("Obj:  %p\n", (void*) obj);')
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


def gen_method_getp(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    retarg = Arg(lasttype)
    if is_scalar(lasttype):
        retarg.pointer = True

    action = "getp"
    layers = get_layers(parts)
    if layers > 0:
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        parts,
        const=False,
        action=action,
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


def gen_method_len(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    assert is_array(lasttype)

    retarg = Arg(Int64)

    action = "len"
    layers = get_layers(parts)
    if layers > 0 and not is_scalar(lasttype):
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        parts,
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
        lst.append(gen_method_offset(parts, conf))
        arrarg = Arg(Int64, pointer=True)
        pointed = gen_c_pointed(arrarg, conf)
        lst.append(f"  int64_t* arr= {pointed};")
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


def gen_method_size(cls, parts, conf):
    lasttype = get_inner_type(parts[-1])
    retarg = Arg(Int64)

    action = "size"
    layers = get_layers(parts)
    if layers > 0 and not is_scalar(lasttype):
        action += str(layers)

    kernel = gen_fun_data(
        cls,
        parts,
        const=False,
        action=action,
        extra=[],
        ret=retarg,
    )
    decl = gen_c_decl_from_kernel(kernel, conf)

    lst = [decl + "{"]

    if lasttype._size is None:
        lst.append(gen_method_offset(parts, conf))
        pointed = gen_c_pointed(retarg, conf)
        lst.append(f"  return {pointed};")
    else:
        lst.append(f"  return {lasttype._size};")
    lst.append("}")
    return "\n".join(lst), kernel


def gen_method_shape(cls, parts, conf):
    "return shape in an array"
    return None, None


def gen_method_nd(cls, parts, conf):
    "return length of shape"
    return None, None


def gen_method_strides(cls, parts, conf):
    "return strides"
    return None, None


def gen_method_getpos(cls, parts, conf):
    "return slot position from index and strides"
    return None, None


def gen_typedef(cls):
    typename = cls._c_type
    return f"typedef struct {typename} * {typename};"


def gen_typedef_decl(cls):
    # TODO: moce to class methods
    out = []
    typename = cls._c_type
    if is_struct(cls) or is_array(cls) or is_single_ref(cls):
        out.append(f"#ifndef XOBJ_TYPEDEF_{typename}")
        out.append(f"{gen_typedef(cls)}")
        out.append(f"#define XOBJ_TYPEDEF_{typename}")
        out.append("#endif")
    elif is_ref(cls):
        out.append(f"#ifndef XOBJ_TYPEDEF_{typename}")
        # defining C union might have issues with GPU qualifiers
        out.append(f"{gen_typedef(cls)}")
        lst = ",".join(tt._c_type for tt in cls._rtypes)
        out.append(f"enum {{{lst}}};")
        out.append(f"#define XOBJ_TYPEDEF_{typename}")
        out.append("#endif")
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
    types = {cls._c_type: cls}
    out = []
    for parts in specs:
        for part in parts:
            lasttype = get_inner_type(part)
            types[lasttype._c_type] = lasttype
    for nn, tt in types.items():
        if not is_scalar(tt):
            out.append(gen_typedef(tt))
    return "\n".join(out)


def gen_code(cls, specs, conf):
    out = []
    for parts in specs:
        out.append(gen_method_getp(cls, parts, conf))
        lasttype = get_inner_type(parts[-1])

        if is_scalar(lasttype):
            out.append(gen_method_get(cls, parts, conf))
            out.append(gen_method_set(cls, parts, conf))
        else:
            out.append(gen_method_size(cls, parts, conf))
        if is_array(lasttype):
            out.append(gen_method_len(cls, parts, conf))
            out.append(gen_method_shape(cls, parts, conf))
            out.append(gen_method_nd(cls, parts, conf))
            out.append(gen_method_strides(cls, parts, conf))
            out.append(gen_method_getpos(cls, parts, conf))

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

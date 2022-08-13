# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from collections import defaultdict
from typing import NamedTuple, List, Type
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import weakref
import os

"""

TODO:
    - add test get_a_buffer
    - Consider exposing Buffer and removing CLBuffer, ByteArrayBuffers..
    - Consider Buffer[offset] to create View and avoid _offset in type API
"""


log = logging.getLogger(__name__)


def topological_sort(source):
    """Sort tasks defined as {child: parents} in a list"""

    # Store parent count and parent dictionary.
    graph = {}
    num_parents = defaultdict(int)
    for child, parents in source.items():
        for parent in parents:
            graph.setdefault(parent, []).append(child)
            num_parents[child] += 1

    # Begin with the parent-less items.
    result = [child for child, parents in source.items() if len(parents) == 0]
    result.extend([item for item in graph if num_parents[item] == 0])

    # Descend through graph, removing parents as we go.
    for parent in result:
        if parent in graph:
            for child in graph[parent]:
                num_parents[child] -= 1
                if num_parents[child] == 0:
                    result.append(child)
            del graph[parent]

    # If there's a cycle, just throw in whatever is left over.
    has_cycle = bool(graph)
    if has_cycle:
        result.extend(list(graph.keys()))
    return result, has_cycle


def sort_classes(classes):
    cdict = {cls.__name__: cls for cls in classes}
    deps = {}
    lst = list(classes)
    for cls in lst:
        cldeps = []
        cllist=[]
        if hasattr(cls, "_get_inner_types"):
            cllist.extend(cls._get_inner_types())
        if hasattr(cls, "_depends_on"):
            cllist.extend(cls._depends_on)
        for cl in cllist:
            if not cl.__name__ in cdict:
                cdict[cl.__name__] = cl
                lst.append(cl)
            cldeps.append(cl.__name__)
        deps[cls.__name__] = cldeps
    lst, has_cycle = topological_sort(deps)
    if has_cycle:
        raise ValueError("Class dependencies have cycles")
    return [cdict[cn] for cn in lst if hasattr(cdict[cn], "_gen_c_api")]


def sources_from_classes(classes):
    sources = []
    for cls in classes:
        sources.append(cls._gen_c_api())
        if hasattr(cls, "_extra_c_sources"):
            sources.extend(cls._extra_c_sources)
    return sources


def classes_from_kernels(kernels):
    classes = set()
    for _, kernel in kernels.items():
        classes.update(kernel.get_classes())
    return classes


def _concatenate_sources(sources, apply_to_source=()):
    source = []
    folders = set()
    for ss in sources:
        if isinstance(ss, Source):
            ss = ss.source

        if hasattr(ss, "read"):
            source.append(ss.read())
            folders.add(os.path.dirname(ss.name))
        elif isinstance(ss, Path):
            with open(ss, "r") as fid:
                source.append(fid.read())
            folders.add(ss.parent)
        else:
            source.append(ss)
    source = "\n".join(source)

    folders = [str(ff) for ff in folders]

    for ff in apply_to_source:
        source = ff(source)

    return source, folders


def _align(offset, alignment):
    "round to nearest multiple of 8"
    return (offset + alignment - 1) & (-alignment)


class MinimalDotDict(dict):
    def __getattr__(self, attr):
        if attr not in self:
            raise AttributeError(f"`{attr}` not found in dict")
        return self.get(attr)

    def __dir__(self):
        return list(self.keys())


class ModuleNotAvailable(object):
    def __init__(self, message="Module not available"):
        self.message = message

    def __getattr__(self, attr):
        raise NameError(self.message)


class XContext(ABC):
    minimum_alignment = 1

    def __init__(self):
        self._kernels = MinimalDotDict()
        self._buffers = []

    def new_buffer(self, capacity=1048576):
        buf = self._make_buffer(capacity=capacity)
        self.buffers.append(weakref.finalize(buf, log.debug, "free buf"))
        return buf

    @property
    def buffers(self):
        return self._buffers

    @property
    def kernels(self):
        return self._kernels

    @abstractmethod
    def _make_buffer(self, capacity):
        "return buffer"

    @abstractmethod
    def add_kernels(
        self,
        sources: list,
        kernels: dict,
        specialize: bool,
        apply_to_source: list,
        save_source_as: str,
    ):
        pass

    @abstractmethod
    def nparray_to_context_array(self, arr):
        return arr

    @abstractmethod
    def nparray_from_context_array(self, dev_arr):
        return dev_arr

    @property
    @abstractmethod
    def nplike_lib(self):
        "return lib"

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def zeros(self, *args, **kwargs):
        "return arr"

    @abstractmethod
    def plan_FFT(self, data, axes):
        "return fft"


class XBuffer(ABC):
    def __init__(
        self,
        capacity=1048576,
        context=None,
        default_alignment=None,
        grow_step=None,
    ):

        if context is None:
            self.context = self._make_context()
        else:
            self.context = context
        self.buffer = self._new_buffer(capacity)
        self.capacity = capacity
        if default_alignment is None:
            default_alignment = self.context.minimum_alignment
        self.default_alignment = default_alignment
        self.chunks = [Chunk(0, capacity)]
        self.grow_step = grow_step

    @abstractmethod
    def _make_context(self):
        "return a default context"

    def allocate(self, size, align=True):
        # find available free slot
        # and update free slot if exists
        if align:
            alignment = self.default_alignment
        else:
            alignment = 1
        # sizepa = size + alignment - 1
        for chunk in self.chunks:
            offset = _align(chunk.start, alignment)
            newend = offset + size
            if chunk.end >= newend:
                chunk.start = newend
                if chunk.size == 0:
                    self.chunks.remove(chunk)
                return offset

        # no free slot check if can be allocated then try to grow
        sizepa = size + alignment - 1
        if sizepa > self.capacity:
            self.grow(sizepa)
        elif self.grow_step is not None:
            self.grow(self.grow_step)
        else:
            self.grow(self.capacity)

        # try again
        return self.allocate(size, align=align)

    def grow(self, capacity):
        """
        Add capacity to buffer
        """
        oldcapacity = self.capacity
        newcapacity = self.capacity + capacity
        newbuff = self._new_buffer(newcapacity)
        self.copy_to_native(
            dest=newbuff, dest_offset=0, source_offset=0, nbytes=oldcapacity
        )
        self.buffer = newbuff
        if len(self.chunks) == 0 or self.chunks[-1].end != self.capacity:
            self.chunks.append(Chunk(oldcapacity, newcapacity))
        else:  # free chunk is at the end
            self.chunks[-1].end = newcapacity

        self.capacity = newcapacity

    def free(self, offset, size):
        nch = Chunk(offset, offset + size)
        # insert sorted
        if offset > self.chunks[-1].start:  # new chuck at the end
            self.chunks.append(nch)
        else:  # new chuck needs to be inserted
            for ic, ch in enumerate(self.chunks):
                if offset <= ch.start:
                    self.chunks.insert(ic, nch)
                    break
        # merge chunks
        pch = self.chunks[0]
        newchunks = [pch]
        for ch in self.chunks[1:]:
            if pch.overlaps(ch):
                pch.merge(ch)
            else:
                newchunks.append(ch)
                pch = ch
        self.chunks = newchunks

    @abstractmethod
    def _new_buffer(self, capacity):
        "return newbuffer"

    @abstractmethod
    def update_from_native(self, offset, source, source_offset, nbytes):
        """Copy data from native buffer into self.buffer starting from offset"""

    @abstractmethod
    def copy_to_native(self, dest, dest_offset, source_offset, nbytes):
        """copy data from self.buffer into dest"""

    @abstractmethod
    def copy_native(self, offset, nbytes):
        """return native data with content at from offset and nbytes"""

    @abstractmethod
    def update_from_buffer(self, offset, source):
        """Copy data from python buffer such as bytearray, bytes, memoryview, numpy array.data"""

    @abstractmethod
    def to_nplike(self, offset, dtype, shape):
        """view in nplike"""

    @abstractmethod
    def update_from_nplike(self, offset, dest_dtype, value):
        """update data from nplike matching dest_dtype"""

    @abstractmethod
    def to_bytearray(self, offset, nbytes):
        """copy in byte array: used in update_from_xbuffer"""

    @abstractmethod
    def to_pointer_arg(self, offset, nbytes):
        """return data that can be used as argument in kernel"""

    def update_from_xbuffer(self, offset, source, source_offset, nbytes):
        """update from any xbuffer, don't pass through gpu if possible"""
        if source.context == self.context:
            self.update_from_native(
                offset, source.buffer, source_offset, nbytes
            )
        else:
            data = source.to_bytearray(source_offset, nbytes)
            self.update_from_buffer(offset, data)

    def get_free(self):
        return sum([ch.size for ch in self.chunks])

    def __repr__(self):
        name = self.__class__.__name__
        return f"<{name} {self.get_free()}/{self.capacity}>"


class Chunk:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def size(self):
        return self.end - self.start

    #    def overlaps(self,other):
    #        return not ((other.end < self.start) or (other.start > self.end))

    def overlaps(self, other):
        return (other.end >= self.start) and (other.start <= self.end)

    def merge(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        return self

    def copy(self):
        return Chunk(self.start, self.end)

    def __repr__(self):
        return f"Chunk({self.start},{self.end})"


class View(NamedTuple):
    buffer: XBuffer
    offset: int
    size: int


available: List[Type[XContext]] = []


class Arg:
    def __init__(
        self, atype, pointer=False, name=None, const=False, factory=None
    ):
        self.atype = atype
        self.pointer = pointer
        self.name = name
        self.const = const
        self.factory = factory

    def get_c_type(self):
        ctype = self.atype._c_type
        if self.pointer:
            ctype += "*"
        return ctype


class Kernel:
    def __init__(self, args, c_name=None, ret=None, n_threads=1):
        self.c_name = c_name
        self.args = args
        self.ret = ret
        self.n_threads = n_threads

    def get_classes(self):
        classes = [
            a.atype for a in self.args if hasattr(a.atype, "_gen_c_api")
        ]
        if isinstance(self.ret, Arg) and hasattr(self.ret.atype, "_gen_c_api"):
            classes.append(self.ret.atype)
        return classes

class Source:
    def __init__(self, source, name=None):
        self.source = source
        self.name = name

class Method:
    def __init__(self, args, c_name, ret):
        self.args = args
        self.c_name = c_name
        self.ret = ret

    def mk_method(self):
        def a_method(instance, *args, **kwargs):
            context = instance._buffer.context
            kernel = context.kernels[self.kernel_name]
            kwargs[self.arg_name] = instance
            return kernel(*args, **kwargs)

        return a_method


def get_context_from_string(ctxstr):
    import xobjects as xo

    if ctxstr is None:
        return xo.ContexCPU()
    else:
        ll = ctxstr.split(":")
        if len(ll) <= 1:
            ctxtype = ll[0]
            option = []
        else:
            ctxtype, options = ctxstr.split(":")
            option = options.split(",")
    if ctxtype == "ContextCpu":
        if len(option) == 0:
            return xo.ContextCpu()
        else:
            return xo.ContextCpu(omp_num_threads=int(option[0]))
    elif ctxtype == "ContextCupy":
        if len(option) == 0:
            return xo.ContextCupy()
        else:
            return xo.ContextCupy(device=int(option[0]))
    elif ctxtype == "ContextPyopencl":
        if len(option) == 0:
            return xo.ContextPyopencl()
        else:
            return xo.ContextPyopencl(device=option[0])
    else:
        raise ValueError(f"Cannot create context from `{ctxstr}`")


def get_test_contexts():
    import os
    import xobjects as xo

    ctxstr = os.environ.get("XOBJECTS_TEST_CONTEXTS")
    if ctxstr is None:
        yield xo.ContextCpu()
        #yield xo.ContextCpu(omp_num_threads=2)
        if xo.ContextCupy in xo.context.available:
            yield xo.ContextCupy()
        if xo.ContextPyopencl in xo.context.available:
            yield xo.ContextPyopencl()
    elif ctxstr == "all":
        yield xo.ContextCpu()
        yield xo.ContextCpu(omp_num_threads=2)
        if xo.ContextCupy in xo.context.available:
            yield xo.ContextCupy()
        if xo.ContextPyopencl in xo.context.available:
            for dd in xo.ContextPyopencl.get_devices():
                yield xo.ContextPyopencl(device=dd)

    else:
        for cc in ctxstr.split(";"):
            yield get_context_from_string(cc)


def get_user_context():
    """
    Get the context specfied by the enviroment variable XOBJECTS_USER_CONTEXT.
    If not present use ContextCpu().

    Examples:
       ContextPyopencl:0.0  -> ContextPyopencl(device="0.0")
       ContextPyopencl:1.0  -> ContextPyopencl(device="1.0")
       ContextPyopencl      -> ContextPyopencl()
       ContextCpu           -> ContextCpu()
       ContextCpu:2         -> ContextCpu(omp_num_threads=2)
       ContextCupy:0        -> ContextCupy(device=0)
    """
    import os

    ctxstr = os.environ.get("XOBJECTS_USER_CONTEXT")
    return get_context_from_string(ctxstr)

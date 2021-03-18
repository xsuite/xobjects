from typing import NamedTuple, Bool

class field(NamedTuple):
    field_name: str
    type_name: str
    offset: int
    is_deferred: bool
    def get_offset(self):
        return str(self.offset)

class index(NamedTuple):
    strides: list(int)
    offset: int
    type_name: str
    is_deferred: bool
    def get_offset(self):
        return '+'.join(["ii{ii}*{strides[ii]}" for ii in self.strides])

def get_position(specs):
    offset=0
    offlist=[]
    for spec in specs:
        if spec.is_deferred:
            offlist.append(f"  offset+=(int64_t *) obj[offset+spec.get_offset()];")
        else:
            offlist.append(f"  offset+={spec.get_offset()};")
    return '\n'.join(offlist)










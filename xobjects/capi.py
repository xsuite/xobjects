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

# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from xobjects.context import Chunk


def test_chunk():
    a = Chunk(3, 6)
    tests = [Chunk(ss, ee) for ss in range(10) for ee in range(11) if ee > ss]

    def has_no_gaps(lst):
        full = set(range(lst[0], lst[-1]))
        return len(full - set(lst)) == 0

    for b in tests:
        lst = list(range(a.start, a.end))
        lst += list(range(b.start, b.end))
        lst.sort()
        c = a.copy().merge(b) if a.overlaps(b) else None
        assert a.overlaps(b) == has_no_gaps(lst)
        # print(a,b,a.overlaps(b),has_no_gaps(lst),c)

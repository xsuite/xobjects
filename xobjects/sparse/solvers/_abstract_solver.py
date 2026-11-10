# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from abc import ABC, abstractmethod

class SuperLUlikeSolver(ABC):

    @abstractmethod
    def __init__(self, A):
        pass

    @abstractmethod
    def solve(self,b):
        return x
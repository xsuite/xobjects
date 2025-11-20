from abc import ABC, abstractmethod

class SuperLUlikeSolver(ABC):

    @abstractmethod
    def __init__(self, A):
        pass

    @abstractmethod
    def solve(self,b):
        return x
from abc import ABC, abstractmethod

class TaskDriver(ABC):
    def __init__(self):
        pass

    def setup_env(self):
        pass

    """
        Abstract methods are given a qcelement molecule as input.
    """

    @abstractmethod
    def energy(self, mol):
        pass
    
    @abstractmethod
    def gradient(self, mol):
        pass
    
    @abstractmethod
    def hessian(self, mol):
        pass
    



from abc import ABC, abstractmethod
from qcelemental.models.v2.molecule import Molecule

class TaskDriver(ABC):
    def __init__(self):
        pass

    def setup_env(self):
        pass

    """
        Abstract methods are given a qcelement molecule as input.
    """

    @abstractmethod
    def energy(self, mol: Molecule):
        pass
    
    @abstractmethod
    def gradient(self, mol: Molecule):
        pass
    
    @abstractmethod
    def hessian(self, mol: Molecule):
        pass
    



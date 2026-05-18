import pytest
import scipy
import qcelemental as qcel
from mrtunneling.task_driver import TaskDriver
from mrtunneling.bead import Bead
from mrtunneling.ring_polymer import RingPolymer

class H2O_TD(TaskDriver):
    """
        H2O harmonic PES assuming O H H atom order.
    """
    def __init__(self):
        super().__init__()
    
    def PES(self, mol_coords):
        r1 = (mol_coords[1,:] - mol_coords[0,:])**2
        r2 = (mol_coords[2,:] - mol_coords[0,:])**2
        r3 = (mol_coords[2,:] - mol_coords[1,:])**2
        return (r1 - 0.96)**2 + (r2 - 0.96)**2 + (r3 - 2.2)**2

    def energy(self, mol):
        return self.PES(mol.geometry)

    def gradient(self, mol):
        return scipy.differentiate.jacobian(self.PES, mol.coords)

    def hessian(self, mol):
        pass

beads = []

def test_rp_init():
    pass

def test_rp_U():
    pass

def test_rp_grad():
    pass

def test_rp_hess():
    pass

def test_rp_evaluate_all_beads():
    pass

def test_rp_align_beads():
    pass

def test_rp_double_beads():
    pass


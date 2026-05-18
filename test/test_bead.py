import pytest
import numpy as np
import qcelemental as qcel
from mrtunneling.task_driver import TaskDriver
from mrtunneling.bead import Bead

class FauxTaskDriver(TaskDriver):
    def energy(self, mol):
        return 0.0
    def gradient(self, mol):
        return np.zeros(mol.geometry.shape)
    def hessian(self, mol):
        return np.zeros((len(mol.geometry.flatten()), len(mol.geometry.flatten())))

ftd = FauxTaskDriver()

mol = qcel.models.Molecule.from_data({
    "symbols": ["H","H"],
    "geometry": np.array([[0.4,0.0,0.0],[-0.4,0.0,0.0]])
})

def test_bead_energy():
    b = Bead(mol)
    assert b.has_V is False
    assert b.has_data is False
    with pytest.raises(AttributeError):
        b.energy
    b.energy = ftd.energy
    assert b.has_V is True
    assert b.has_data is True
    assert b.energy == 0.0

def test_bead_gradient():
    b = Bead(mol)
    assert b.has_grad is False
    assert b.has_data is False
    with pytest.raises(AttributeError):
        b.gradient
    b.gradient = ftd.gradient
    assert b.has_grad is True
    assert b.has_data is True
    assert np.isclose(b.gradient, np.zeros(len(b))).all()

def test_bead_hessian():
    b = Bead(mol)
    assert b.has_hess is False
    assert b.has_data is False
    with pytest.raises(AttributeError):
        b.hessian
    b.hessian = ftd.hessian
    assert b.has_hess is True
    assert b.has_data is True
    assert np.isclose(b.hessian, np.zeros((len(b), len(b)))).all()


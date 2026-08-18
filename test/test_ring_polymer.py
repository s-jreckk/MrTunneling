import pytest
import numpy as np
import scipy
import qcelemental as qcel
from mrtunneling.constants import hbar
from mrtunneling.task_driver import TaskDriver
from mrtunneling.bead import Bead
from mrtunneling.ring_polymer import RingPolymer

mol = qcel.models.Molecule.from_data({"symbols": ["H", "C", "O"], "geometry": np.eye(3)})
grad_scale = 1e-5
hess_scale = 1e-5

class TD(TaskDriver):
    def __init__(self):
        super().__init__()
    
    def energy(self, mol):
        return 0.0

    def gradient(self, mol):
        return grad_scale * np.ones((3,3))

    def hessian(self, mol):
        return hess_scale * np.ones((9,9))

td = TD()
b1 = Bead(mol)
b2 = Bead(mol)
b2.mol = np.array([[0,-1,0],[1,0,0],[0,0,1.2]])
beads = [b1, b2]

def test_rp_init():
    rp = RingPolymer(298.15, beads, td)
    assert rp.N == 4
    assert np.isclose(rp.beta, 1059.114623)
    assert np.isclose(rp.betaN, 264.7786557)

def test_rp_align_beads():
    rp = RingPolymer(298.15, beads, td)
    [print(i) for i in rp.beads]
    rp.align_beads()
    aligned_geom = np.array([[ 1.022874692189,  0.022874689785, -0.045749379571],
                             [ 0.022874689785,  1.022874692189, -0.045749379571],
                             [-0.045749379571, -0.045749379571, 1.091498761545]])
    assert np.isclose(rp.beads[1].mol.geometry, aligned_geom).all()

@pytest.mark.parametrize("d", [{"der_lvl":1}, {"der_lvl":2}])
def test_rp_evaluate_all_beads(d):
    rp = RingPolymer(298.15, beads, td)
    rp.align_beads()
    rp.evaluate_all_beads(der_lvl=d["der_lvl"], update_hess=False)
    assert rp.exact_hessian == (d["der_lvl"] > 1)
    for b in rp.beads:
        assert b.has_V
        assert np.isclose(b.energy, 0.0)
        assert b.has_grad
        assert np.isclose(b.gradient, grad_scale*np.ones((3,3)).flatten()).all()
        assert b.has_hess == (d["der_lvl"] > 1)
        if d["der_lvl"] > 1:
            assert np.isclose(b.hessian, hess_scale*np.ones((9,9))).all()

def test_rp_U():
    rp = RingPolymer(298.15, beads, td)
    rp.align_beads()
    rp.evaluate_all_beads(der_lvl=0, update_hess=False)
    u = rp.U()
    assert np.isclose(u, 1.723796022E-06)

def test_rp_grad():
    rp = RingPolymer(298.15, beads, td)
    rp.align_beads()
    rp.evaluate_all_beads(der_lvl=1, update_hess=False)
    g = rp.gradient()
    grad_base = np.array([
        [ 3.288323e-07,  3.288323e-07, -6.576645e-07],
        [ 3.915349e-06,  3.915350e-06, -7.830699e-06],
        [-1.043761e-05, -1.043761e-05,  2.087523e-05]]).flatten()
    grad0 = -grad_base + grad_scale
    grad1 = grad_base + grad_scale
    assert np.isclose(grad0, g[:len(b1)]).all()
    assert np.isclose(grad1, g[len(b1):]).all()

def test_rp_hess():
    rp = RingPolymer(298.15, beads, td)
    rp.align_beads()
    rp.evaluate_all_beads(der_lvl=2)
    h = rp.hessian()

    diag_bit = np.diag(np.repeat(mol.masses, 3))
    diag_bit *= 1.0 / (264.7786557**2 * hbar**2)

    d = hess_scale * np.ones((9,9)) + diag_bit
    offd = -diag_bit
    answer = np.block([[d,offd],[offd,d]]) 
    print(h)
    print(answer)
    assert np.isclose(h, answer).all()

#def test_rp_double_beads():
#    pass


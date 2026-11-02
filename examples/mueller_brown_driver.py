import numpy as np
import math
from qcelemental.models.v2 import Molecule

from mrtunneling import TaskDriver

class MuellerBrownRunner(TaskDriver):

    Parameters = {
        "A": [-200.0, -100.0, -170.0, 15.0],
        "a": [-1.0, -1.0, -6.5, 0.7],
        "b": [0.0, 0.0, 11.0, 0.6],
        "c": [-10.0, -10.0, -6.5, 0.7],
        "x": [1.0, 0.0, -0.5, -1.0],
        "y": [0.0, 0.5, 1.5, 1.0]
    }

    def __init__(self):
        super().__init__()

    def energy(self, mol: Molecule):
        x = mol.geometry[0][0]
        y = mol.geometry[0][1]
        z = mol.geometry[0][2]
        return MuellerBrownRunner.en(x, y, z)

    @staticmethod
    def en(x, y, z):
        p = MuellerBrownRunner.Parameters
        v = 5000*z**2
        for k in range(4):
            v += p["A"][k] * math.exp(MuellerBrownRunner.g(x, y, k))
        return v

    def gradient(self, mol):
        x = mol.geometry[0][0]
        y = mol.geometry[0][1]
        z = mol.geometry[0][2]
        print(x,y,z)
        return MuellerBrownRunner.grad(x ,y, z)

    @staticmethod
    def grad(x, y, z):
        p = MuellerBrownRunner.Parameters
        gx = 0
        gy = 0
        gz = 10000*z
        for k in range(4):
            gx += p["A"][k] * math.exp(MuellerBrownRunner.g(x, y, k)) * (2*p["a"][k]*(x-p["x"][k]) + p["b"][k]*(y-p["y"][k]))
            gy += p["A"][k] * math.exp(MuellerBrownRunner.g(x, y, k)) * (2*p["c"][k]*(y-p["y"][k]) + p["b"][k]*(x-p["x"][k]))
        return np.array([gx, gy, gz])


    def hessian(self, mol, h=1e-5):
        x = mol.geometry[0][0]
        y = mol.geometry[0][1]
        z = mol.geometry[0][2]
        return MuellerBrownRunner.hess(x, y, z, h)

    @staticmethod
    def hess(x, y, z, h):
        H = np.zeros([3,3])
        H[0] = (MuellerBrownRunner.grad(x + h, y, z) - MuellerBrownRunner.grad(x - h, y, z))
        H[1] = (MuellerBrownRunner.grad(x, y + h, z) - MuellerBrownRunner.grad(x, y - h, z))
        H[2] = (MuellerBrownRunner.grad(x, y, z + h) - MuellerBrownRunner.grad(x, y, z - h))
        return H / (2*h)

    @staticmethod
    def g(x, y, k):
        p = MuellerBrownRunner.Parameters
        return p["a"][k]*(x-p["x"][k])**2 + p["b"][k]*(x-p["x"][k])*(y-p["y"][k]) + p["c"][k]*(y-p["y"][k])**2

    def energy_and_gradient(self, mol):
        x = mol.geometry[0][0]
        y = mol.geometry[0][1]
        z = mol.geometry[0][2]
        return MuellerBrownRunner.en(x, y, z), MuellerBrownRunner.grad(x, y, z)
import psi4
from qcelemental.models.v2 import Molecule
import matplotlib.pyplot as plt
import numpy as np

import mueller_brown_driver as mbd

runner = mbd.MuellerBrownRunner()

x = -0.822
y =  0.624
z =  0
molstr = "O " + str(x*psi4.constants.bohr2angstroms) + " " + str(y*psi4.constants.bohr2angstroms) + " " + str(z*psi4.constants.bohr2angstroms)

mol = Molecule.from_data(molstr)

E = runner.energy(mol)
G = runner.gradient(mol)
H = runner.hessian(mol)

print(E, G, "\n\n", H)

fcs, modes = np.linalg.eigh(H)
print("\n\n", fcs, modes)

x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)

# Evaluate scalar function over grid
Z = np.vectorize(runner.en)(X, Y, 0)

Zlog = np.log(Z + 180)

# Choose displayed range
vmin = 3.0
vmax = 8.0

# More contour levels near low values
t = np.linspace(0, 1, 50)
levels = vmin + (vmax - vmin) * t**5

plt.contourf(
    X, Y, Zlog,
    levels=levels,
    cmap='viridis',
    extend='max'
)

plt.colorbar(label='log(Z - Zmin)')
plt.show()
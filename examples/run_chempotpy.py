from chempotpy_driver import ChemPotPyRunner
from qcelemental.models.v2.molecule import Molecule

import numpy as np
import matplotlib.pyplot as plt

distances = np.linspace(0.0, 3.0, 100)
energies = []

o3runner = ChemPotPyRunner("O3", "CH5_GEN_J2_1987")

for r in distances:

    geom = f"""
        O   0.0     0.0    -0.1
        O   0.0     0.0     {r}
        O   0.0     0.0     3.1
    """

    e = o3runner.energy(Molecule.from_data(geom))

    energies.append(np.asarray(e).squeeze())

plt.plot(distances, energies)
plt.xlabel("Position of O atom / Å")
plt.ylabel("Potential energy")
plt.title("O3 potential-energy scan")

plt.show()
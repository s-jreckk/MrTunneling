import numpy as np
from copy import deepcopy

from .constants import *
from .utils import proc_Hess
from .ring_polymer import RingPolymer
from .bead import Bead

class Instanton(RingPolymer):
    def __init__(self, wb, T, beads, task_driver):
        self.wb = wb
        super().__init__(T, beads, task_driver)

    def crossover_T(self):
        return hbar * np.abs(self.wb) / (2*np.pi*kb)

    def update(self):
        pass

    def add_beads(self):
        pass

    def check_convergence(self):
        pass

    @classmethod
    def initiate_from_TS(cls, T, TSmol, TShess, Nbeads, task_driver):
        # Do Hessian analysis on TShess
        freqs, nmodes = proc_Hess(TSmol, TShess)
        if freqs[0] > 0.0:
            raise ValueError("Imaginary mode has positive frequency!")
        print(f"Selecting mode with frequency {freqs[0]*h2cm:5.2f} cm-1")
        qi = nmodes[:,0] # Imaginary freq mode is assumed to be the largest negative freq
        geoms = []
        delta = 0.2
        # Make half of ring polymer, the other half wraps back around and is equivalent
        # Nbeads = N/2
        for i in range(Nbeads):
            pf = delta * np.cos(np.pi*i/(Nbeads-1)) # Go from +q_i to -q_i once
            newgeom = TSmol.geometry + pf * qi.reshape(*TSmol.geometry.shape)
            mol_i = TSmol.copy(update={"geometry": newgeom})
            geoms.append(mol_i)
        beads = [Bead(g) for g in geoms]
        return cls(freqs[0], T, beads, task_driver)

    def write_to_mXYZ(self, fn):
        with open(f"{fn}", "w") as f:
            f.write("".join([str(i) for i in self.beads]))
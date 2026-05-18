import numpy as np
from .constants import kb, hbar
from .task_driver import TaskDriver

class RingPolymer():
    def __init__(self, T, beads, task_driver:TaskDriver):
        self.beads = beads # Bead 1 to N/2, indexed minus 1 bc Python
        self.N = len(self.beads) * 2
        self.T = T
        self.beta = (kb*self.T)**-1
        self.betaN = self.beta / self.N
        self.exact_hessian = False
        self.task_driver = task_driver

    def U(self):
        out = 0.0
        for b, bead in enumerate(self.beads): # Sum from 1 to N
            if b == 0:
                continue
            Vx = bead.V
            d = bead.dist(self.beads[b-1]) # f x 3 matrix
            d *= d
            Vharm = np.sum(d * bead.mol.masses[:,None])
            Vharm /= 2*(self.betaN**2)*(hbar**2)
            out += Vx + Vharm
        return out

    def gradient(self):
        # Gradient of R.P.
        prefactor = np.repeat(self.beads[0].mol.masses, 3) / (self.betaN**2 * hbar**2)
        big_grad = np.array([])
        for n in range(len(self.beads)):
            # n idxs each bead of N/2 polymer
            lil_grad = np.copy(self.beads[n].gradient).flatten()
            if n == 1:
                lil_grad += prefactor * (self.beads[0].mol.geometry - self.beads[1].mol.geometry).flatten()
            elif n == len(self.beads) - 1:
                lil_grad += prefactor * (self.beads[n].mol.geometry - self.beads[n-1].mol.geometry).flatten()
            else:
                lil_grad += prefactor * 2.0*self.beads[n].mol.geometry.flatten() 
                lil_grad -= prefactor * self.beads[n-1].mol.geometry.flatten() 
                lil_grad -= prefactor * self.beads[n+1].mol.geometry.flatten()
            big_grad = np.concatenate((big_grad, lil_grad))
        return big_grad
    
    def hessian(self):
        # Hessian of R.P.
        pass
    
    def evaluate_all_beads(self, do_hess=False):
        for bi, b in enumerate(self.beads):
            b.energy = self.task_driver.energy
            b.gradient = self.task_driver.gradient
            print(f"Bead {bi} energy and gradient computed.")
            if do_hess:
                b.hessian = self.task_driver.hessian
                print(f"Bead {bi} energy and gradient computed.")
                self.exact_hessian = True
            else:
                self.hessian_update()
                self.exact_hessian = False

    def hessian_update(self):
        pass

    def double_beads(self):
        pass

    def align_beads(self):
        for b in range(1,len(self.beads)):
            if self.beads[b].has_data:
                print(Warning("Aligning beads with computed data will overwrite the computed data."))
            self.beads[b].mol = self.beads.mol[b].align(
                self.beads.mol[b-1], atoms_map=True)


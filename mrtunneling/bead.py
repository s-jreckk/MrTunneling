import pathlib
import numpy as np
from .constants import au_to_amu, ref_mass

class Bead():
    def __init__(self, mol):
        self._mol = mol
        self._V = None
        self._grad = None
        self._hess = None
        self._oldmol = None
        self._oldV = None
        self._oldgrad = None
        self._oldhess = None
        self.has_V = False
        self.has_grad = False
        self.has_hess = False
        self.exact_hessian = False
        self.masses = np.repeat(self.mol.masses, 3) / au_to_amu # 3N Masses in m_e
        self.natoms = len(self.mol.masses)
        self.M    = np.diag(np.sqrt(self.masses / ref_mass))
        self.Minv = np.diag(np.sqrt(self.masses / ref_mass)**-1)

    def __str__(self):
        s = self._mol.to_string("xyz").splitlines()
        # Put energy in comment line
        if self.has_V:
            s[1] = f"   {self.energy:14.10f}"
        else:
            s[1] = f"   {0.0}"
        s = "\n".join(s)
        return s

    def __len__(self):
        return len(self._mol.geometry.flatten())

    @property
    def x(self):
        #return self.M @ self.mol.geometry.flatten()
        return self.mol.geometry.flatten()

    @property
    def mol(self):
        return self._mol

    def update(self, newmol_geom):
        if not self.has_V or not self.has_grad or not self.has_hess:
            raise AttributeError("Energy, gradient, and/or Hessian has not been evaluated prior to bead update.")
        self._oldmol = self._mol.copy()
        self._oldV = self._V
        self._oldgrad = self._grad
        self._oldhess = self._hess
        self.mol = newmol_geom

    @mol.setter
    def mol(self, newmol_geom):
        self._mol = self._mol.copy(update={"geometry": newmol_geom})
        self.has_V = False
        self.has_grad = False
        self.has_hess = False

    @property
    def has_data(self):
        return self.has_V or self.has_grad or self.has_hess

    @property
    def energy(self):
        if self.has_V:
            return self._V
        else:
            raise AttributeError("No energy for structure!")

    @energy.setter
    def energy(self, energy_fxn):
        self._V = energy_fxn(self.mol)
        self.has_V = True

    @property
    def gradient(self):
        if self.has_grad:
            return self._grad
        else:
            raise AttributeError("No gradient for structure!")

    @gradient.setter
    def gradient(self, grad_fxn):
        g = grad_fxn(self.mol)
        if len(g.flatten()) != len(self):
            raise ValueError("Computed gradient length does not match Bead molecule.")
        self._grad = g.flatten()
        self.has_grad = True

    @property
    def hessian(self):
        if self.has_hess:
            return self._hess
        else:
            raise AttributeError("No Hessian for structure!")

    @hessian.setter
    def hessian(self, hess_fxn):
        h = hess_fxn(self.mol)
        if h.shape[0] != len(self) or h.shape[1] != len(self):
            raise ValueError("Computed Hessian length does not match Bead molecule.")
        self._hess = h
        self.has_hess = True

    # DFP positive definite
    #def update_hessian(self, mol):
    #    if self._oldhess is None:
    #        raise AttributeError("No Hessian has been previosly computed!")
    #    sk = (self.mol.geometry - self._oldmol.geometry).flatten()
    #    g = self.gradient
    #    gk = self._oldgrad
    #    l = len(self)
    #    yk = g - gk
    #    gamma = np.dot(yk, sk) ** -1
    #    syT = np.outer(sk, yk)
    #    Bkp1 = (np.eye(l) - gamma*syT.T) @ self._oldhess @ (np.eye(l) - gamma * syT) + gamma * np.outer(yk, yk)
    #    return Bkp1
    
    # Bofill as used in Optking
    def update_hessian(self, mol):
        if self._oldhess is None: # Consider using empirical Hess guess
            raise AttributeError("No Hessian has been previosly computed!")
        # Bofill = (1-phi) * MS + phi * Powell
        dq = (self.mol.geometry - self._oldmol.geometry).flatten()
        dg = self.gradient - self._oldgrad
        dqdq = dq @ dq
        H = self._oldhess

        Z = -1.0 * np.dot(H, dq) + dg
        qz = np.dot(dq, Z)
        zz = np.dot(Z, Z)

        phi = 1.0 - qz * qz / (dqdq * zz)
        if phi < 0.0:
            phi = 0.0
        elif phi > 1.0:
            phi = 1.0
        
        MS = np.outer(Z,Z) / qz
        Powell = (dq @ Z / dqdq**2) * np.outer(dq, dq)
        Powell += (1.0 / dqdq) * (np.outer(Z, dq) + np.outer(dq, Z))
        H_new = H + phi * Powell + (1.0 - phi) * MS
        return H_new

    def write(self, dir, idx):
        root = f"bead_{idx}"
        with open(dir / f"{root}.mol", "w") as f:
            f.write(self.__str__())
        if self.has_grad:
            np.savetxt(dir / f"{root}.grad", self._grad)
        if self.has_hess:
            np.savetxt(dir / f"{root}.hess", self._hess)

#    def _check_quantity(self, X):
#        # X is a vector or square matrix with shape 3N or 3Nx3N
#        if X.ndim == 1 or X.ndim == 2:
#            # Check shape
#            if np.array([i==3*self.natoms for i in X.shape]).all():
#                pass
#            else:
#                raise ValueError(f"Cannot mass weight quantity of shape {X.shape}")
#        else:
#            raise ValueError(f"Cannot mass weight quantity of shape {X.shape}")
#
#    def mass_weight_quantity(self, X):
#        self._check_quantity(X)
#        if X.ndim == 1:
#            return self.mW_M @ X
#        elif X.ndim == 2:
#            return self.mW_M @ X @ self.mW_M
#
#    def demass_weight_quantity(self, X):
#        self._check_quantity(X)
#        if X.ndim == 1:
#            return self.dmW_M @ X
#        elif X.ndim == 2:
#            return self.dmW_M @ X @ self.dmW_M




class Bead():
    def __init__(self, mol):
        self._mol = mol

        self._V = None
        self._grad = None
        self._hess = None
        self.has_V = False
        self.has_grad = False
        self.has_hess = False

    def dist(self, other_bead):
        return self.mol.coords - other_bead.mol.coords

    def __str__(self):
        return self._mol.to_string("xyz")

    def __len__(self):
        return len(self._mol.geometry.flatten())

    @property
    def mol(self):
        return self._mol

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


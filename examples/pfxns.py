import qcelemental as qcel
import psi4
from mrtunneling import TaskDriver

class PsiRunner(TaskDriver):
    def __init__(self):
        super().__init__()
        psi4.set_memory('500 MB')
        #psi4.set_output_file('output.dat', False)
        psi4.core.be_quiet()
        psi4.set_options({'reference': 'uhf', "basis": "sto-3g"})
        self.method = "scf"

    def energy(self, mol):
        psi4.geometry(mol.to_string(dtype="psi4"))
        return psi4.energy(self.method)
        
    def gradient(self, mol):
        psi4.geometry(mol.to_string(dtype="psi4"))
        G, wfn = psi4.gradient(self.method, return_wfn=True)
        return G.np

    def hessian(self, mol):
        psi4.geometry(mol.to_string(dtype="psi4"))
        H, wfn = psi4.hessian(self.method, return_wfn=True)
        return H.np



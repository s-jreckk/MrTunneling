import qcelemental as qcel
import numpy as np
from pfxns import PsiRunner
import mrtunneling

pRun = PsiRunner()

# CH4 + H TS HF/STO-3G
molstr = """
0 2

  C            0.103078193976    -0.009035771321    -0.113423145025 
  H            0.608251306929    -0.959486718791     0.006807658474 
  H            0.775003550476     0.831178743166     0.011418788290 
  H           -0.771449723528     0.069805506653     0.848391957082 
  H           -0.453362774775     0.045146679726    -1.041040456325 
  H           -1.385776755607     0.120943170367     1.524932008844 
"""

qcmol = qcel.models.Molecule.from_data(molstr)
E = pRun.energy(qcmol)
G = pRun.gradient(qcmol)
H = pRun.hessian(qcmol)

#freqs, nmodes = mrtunneling.proc_Hess(qcmol, H)

#I.write_to_mXYZ("all.xyz")
#I.evaluate_all_beads()
#g = I.gradient()
#print(g)
#for bead in I.beads:
#    bead.energy = pRun.energy
#    print(bead.energy)

opt_plan = {
        "step_limit": 0.05,
        "max_iter": 50,
        "grad_rms_convergence": 1e-6,
        "grad_max_convergence": 1e-6,
        "step_rms_convergence": 1e-6,
        "step_max_convergence": 1e-6,
        "hess_every": 0,
        "opt_method": "evf"
        }

#I = mrtunneling.Instanton.initiate_from_TS(400.0, qcmol, H, 16, pRun, delta=0.1)
#print(I.crossover_T())
I = mrtunneling.Instanton.restart("step_39", pRun, temp=400.0)
#I.optimize(opt_plan)
#print(I.N, len(I.beads))
#I.double_beads()
#print(I.N, len(I.beads))
#I.optimize(opt_plan)
I.evaluate_all_beads(der_lvl=2)
I.save_state("step_39_exact_hess")
#np.set_printoptions(precision=2, threshold=1000000, linewidth=11900, suppress=True)
print(np.linalg.eigh(I.full_hessian())[0])
#MW_full = np.diag(np.concatenate([np.diag(I.Minv()), np.diag(I.Minv())]))
#MHess = MW_full @ I.full_hessian() @ MW_full
#print(np.sqrt(np.linalg.eigh(MHess)[0]) / mrtunneling.constants.freq_to_hartree)




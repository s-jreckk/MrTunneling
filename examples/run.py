import qcelemental as qcel
import numpy as np
from pfxns import PsiRunner
import mrtunneling

"""
    Example instanton computation for the CH4 + H ---> CH3 + H2 (doublet electronic state) reaction.
    Precomputed are the reactant and transition state (TS) optimized geometries.
    The instanton is initialized from the TS, and then optimized.
    In the example, the optimization completed on step_39.
    Exact Hessians are computed, and the rates determined.

"""

# Define task driver
pRun = PsiRunner()

# Reactant optimized geometry, get energy, gradient, and Hessian for rate computation
# CH4 + H Reactant HF/STO-3G
r_molstr = """
0 2

C     0.185600783674     -0.015759262400     -0.204237642191
H     0.723949795324     -0.952705000257     -0.131943770269
H     0.883125099967      0.808645466307     -0.122182050146
H    -0.541471531280      0.045513716589      0.596089147682
H    -0.323201879001      0.035507345366     -1.158913019022
H    -2.952318219425      0.250681309539      3.248772307704

"""

r_qcmol = qcel.models.Molecule.from_data(r_molstr)
r_E = pRun.energy(r_qcmol)
r_G = pRun.gradient(r_qcmol)
r_H = pRun.hessian(r_qcmol)
# Manually set the necessary values of the bead, I probably need a better way to do this
r_bead = mrtunneling.bead.Bead(r_qcmol)
r_bead._V = r_E
r_bead.grad = r_G
r_bead.has_hess = True
r_bead.exact_hess = True
r_bead._hess = r_H

# Transition state optimized geometry, get energy, gradient, and Hessian for rate computation and instanton initialization
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
# Manually set the necessary values of the bead, I probably need a better way to do this
ts_bead = mrtunneling.bead.Bead(qcmol)
ts_bead._V = E
ts_bead.grad = G
ts_bead.has_hess = True
ts_bead.exact_hess = True
ts_bead._hess = H

# Get frequencies of Hessian (in Hartree)
freqs, nmodes = mrtunneling.proc_Hess(qcmol, H)

# Define plan for optimizing instanton
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

# Initiate instanton from transition state geometry
I = mrtunneling.Instanton.initiate_from_TS(400.0, qcmol, H, 16, pRun, delta=0.1)
# Or restart from directory "step_39"
#I = mrtunneling.Instanton.restart("step_39", pRun, temp=400.0)
I.optimize(opt_plan)

# For final iteration, evaluate exact Hessians
I.evaluate_all_beads(der_lvl=2)
# Save for future use
I.save_state("step_39_exact_hess")
# Or read in if you already have it
#I = mrtunneling.Instanton.read_state("step_39_exact_hess")

# Set up "fake" ring polymers for rate (r) and tunneling correction (k) computations
rRP = mrtunneling.fake_ring_polymer.FakeRingPolymer(I, r_bead)
tsRP = mrtunneling.fake_ring_polymer.FakeRingPolymer(I, ts_bead)

r = mrtunneling.partition_functions.rate(I, rRP)
k = mrtunneling.partition_functions.kappa(I, tsRP)
print(r, k)


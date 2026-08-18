import numpy as np

from .constants import hbar

# I have no clue if any of these work...

def trans_pfxn(ring_polymer):
    M = np.sum(ring_polymer.beads[0].mol.masses)
    out = ((ring_polymer.N*M) / (2*np.pi*ring_polymer.betaN*hbar**2))**(3.0/2.0)
    out /= ring_polymer.N**3
    return out

def rot_pfxn(ring_polymer):
    dI = np.linalg.det(ring_polymer.moit())
    out = np.sqrt(8*np.pi*dI / (ring_polymer.betaN**3 * hbar**6))
    out /= ring_polymer.N**3
    return out

def analyze_hessian(ring_polymer):
    mwHess = ring_polymer.mw_full_hess()
    eigvals, eigvecs = np.linalg.eigh(mwHess)
    # Check Hessian structure
    zeros = np.isclose(eigvals, 0.0, atol=1e-7)
    negatives = eigvals < 0.0
    n_zero = np.sum(zeros)
    if n_zero != 7:
        raise ValueError(f"Wrong number of zero modes {n_zero}.")
    n_negatives = np.sum(negatives & ~zeros)
    if n_negatives != 1:
        raise ValueError(f"Wrong number of negative nonzero modes {n_negatives}.")
    # Return relevant eigenvalues
    return np.sqrt(np.abs(eigvals[~zeros]))

def vib_pfxn_R(ring_polymer, n_ignore=6):
    # Vibrational partition function for collapsed Ring Polymer
    omegas = np.linalg.eigh(ring_polymer.beads[0].hessian)[0][n_ignore:]
    # Solve: sinh(beta_N hbar wt_k / 2) = beta_N hbar w_k / 2
    arcsinh_rhs = np.arcsinh(ring_polymer.betaN * hbar * omegas / 2.0)
    tomegas = 2 * arcsinh_rhs / (ring_polymer.betaN * hbar)
    out = np.prod((2*np.sinh(0.5*ring_polymer.beta*hbar*tomegas))**-1)
    return out

def vib_pfxn_inst(ring_polymer):
    etas = analyze_hessian(ring_polymer)
    beads = ring_polymer.beads
    BN = np.sum([beads[0].masses * (beads[i].x - beads[i-1].x)**2 for i in range(1, len(beads))])
    out = np.prod([1.0/(ring_polymer.betaN * hbar * np.abs(eta)) for eta in etas])
    out *= np.sqrt(2*np.pi*BN / (ring_polymer.betaN * hbar**2))
    out *= ring_polymer.N**7
    return out

def rate(inst_rp, r_rp):
    Q_t = trans_pfxn(inst_rp)
    Q_r = rot_pfxn(inst_rp)
    Q_v = vib_pfxn_inst(inst_rp)
    
def kappa(inst_rp, r_rp):
    Q_r_inst = rot_pfxn(inst_rp)
    Q_v_inst = vib_pfxn_inst(inst_rp)
    Q_r_ts = rot_pfxn(r_rp)
    Q_v_ts = rot_pfxn(r_rp)
    S = 0
    V = 0
    kappa = np.exp((-S/hbar) + (r_rp.beta*V))
    kappa *= (Q_r_inst * Q_v_inst)/(Q_r_ts * Q_v_ts)

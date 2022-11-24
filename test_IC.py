"""
Test obtaining ICs in A from B via LBVP.

Usage:
    c2001_case1.py [options]

Options:
    --Ntheta=<Ntheta>       Latitude coeffs [default: 32]
    --Nr=<Nr>               Radial coeffs  [default: 32]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py', 'matplotlib', 'distributor', 'transforms', 'subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de

def test_IC(Nr, Ntheta):
    L2_error = lambda A, B: de.integ(de.dot(A-B,A-B)).evaluate()['g'][0,0,0]
    L2_set = {}

    # parameters
    Ri = r_inner = 7/13
    Ro = r_outer = 20/13
    Nphi = 2*Ntheta
    dealias = 3/2
    dtype = np.float64

    # Bases
    coords = de.SphericalCoordinates('phi', 'theta', 'r')
    dist = de.Distributor(coords, dtype=dtype)
    basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
    b_inner = basis.S2_basis(radius=r_inner)
    b_outer = basis.S2_basis(radius=r_outer)
    bk1 = basis.clone_with(k=1)
    bk2 = basis.clone_with(k=2)

    # Fields
    A = dist.VectorField(coords, name='A', bases=basis)
    φ = dist.Field(name='φ', bases=bk1)
    τ_A1 = dist.VectorField(coords, name='τ_A1', bases=b_outer)
    τ_A2 = dist.VectorField(coords, name='τ_A2', bases=b_inner)
    τ_φ = dist.Field(name='τ_φ')

    # Substitutions
    phi, theta, r = dist.local_grids(basis)

    lift1 = lambda A, n: de.Lift(A, bk1, n)
    lift = lambda A, n: de.Lift(A, bk2, n)

    ell_func_o = lambda ell: ell+1
    A_potential_bc_o = de.radial(de.grad(A)(r=Ro)) + de.SphericalEllProduct(A, coords, ell_func_o)(r=Ro)/Ro

    ell_func_i = lambda ell: -ell
    A_potential_bc_i = de.radial(de.grad(A)(r=Ri)) + de.SphericalEllProduct(A, coords, ell_func_i)(r=Ri)/Ri

    er = dist.VectorField(coords, bases=basis.radial_basis, name='er')
    er['g'][2] = 1

    # We want to solve for an initial vector potential A
    # with curl(A) = B0, but you're best actually solving -lap(A)=curl(B)
    # (thanks Jeff Oishi & Calum Skene).  We will do this as a BVP.
    B_amp = 5 # 5 is the value in Christensen et al 2001, case 1
    B0 = dist.VectorField(coords, bases=basis)
    B0['g'][0] = B_amp*np.sin(np.pi*(r-Ri))*np.sin(2*theta)
    B0['g'][1] = B_amp/8 * (9*r - 8*Ro - Ri**4/r**3)*np.sin(theta)
    B0['g'][2] = B_amp/8 * (8*Ro - 6*r - 2*Ri**4/r**3)*np.cos(theta)
    J0 = de.curl(B0)

    mag_BVP = de.LBVP([A, φ, τ_A1, τ_A2, τ_φ], namespace=locals())
    mag_BVP.add_equation("-lap(A) + grad(φ) + lift(τ_A1, -1)  + lift(τ_A2, -2)= J0")
    mag_BVP.add_equation("div(A) + τ_φ + lift1(τ_A2,-1)@er = 0")
    mag_BVP.add_equation("A_potential_bc_o = 0")
    mag_BVP.add_equation("A_potential_bc_i = 0")
    mag_BVP.add_equation("integ(φ) = 0")
    solver_BVP = mag_BVP.build_solver()
    solver_BVP.solve()

    label = r'div(A) + τ_φ + lift1(τ_A2,-1)@er'
    B = de.curl(A).evaluate()
    L2_set[label] = L2_error(B0, B)
    print(label, L2_error(B0, B), de.integ(de.div(A)).evaluate()['g'][0,0,0])

    # test old version, with no tau projection in the div(A) equation
    mag_BVP = de.LBVP([A, φ, τ_A1, τ_A2, τ_φ], namespace=locals())
    mag_BVP.add_equation("-lap(A) + grad(φ) + lift(τ_A1, -1)  + lift(τ_A2, -2)= J0")
    mag_BVP.add_equation("div(A) + τ_φ = 0")
    mag_BVP.add_equation("A_potential_bc_o = 0")
    mag_BVP.add_equation("A_potential_bc_i = 0")
    mag_BVP.add_equation("integ(φ) = 0")
    solver_BVP = mag_BVP.build_solver()
    solver_BVP.solve()

    label = r'div(A) + τ_φ'
    B = de.curl(A).evaluate()
    L2_set[label] = L2_error(B0, B)
    print(label, L2_error(B0, B), de.integ(de.div(A)).evaluate()['g'][0,0,0])

    # test an alternative BVP, which directly solves for A from B0
    τ_φ1 = dist.Field(bases=b_outer)
    mag_BVP = de.LBVP([A, φ, τ_A1, τ_φ1, τ_φ], namespace=locals())
    mag_BVP.add_equation("curl(A) + grad(φ) + lift(τ_A1, -1) = B0")
    mag_BVP.add_equation("div(A) + lift(τ_φ1, -1) + τ_φ = 0")
    mag_BVP.add_equation("angular(A_potential_bc_o) = 0", condition='ntheta!=0')
    mag_BVP.add_equation("angular(A_potential_bc_i) = 0", condition='ntheta!=0')
    mag_BVP.add_equation("radial(A_potential_bc_o) = 0", condition='ntheta==0')
    mag_BVP.add_equation("radial(A_potential_bc_i) = 0", condition='ntheta==0')
    mag_BVP.add_equation("integ(φ) = 0")
    solver_BVP = mag_BVP.build_solver()
    solver_BVP.solve()

    label = r'curl(A) + grad(φ) = B0'
    B = de.curl(A).evaluate()
    L2_set[label] = L2_error(B0, B)

    print(label, L2_error(B0, B), de.integ(de.div(A)).evaluate()['g'][0,0,0])

    return L2_set

if __name__=='__main__':
    import matplotlib.pyplot as plt

    from docopt import docopt
    args = docopt(__doc__)

    Nr_max = int(args['--Nr'])
    Ntheta_max = int(args['--Ntheta'])

    def log_set(N_min, N_max):
        log2_N_min = int(np.log2(N_min))
        log2_N_max = int(np.log2(N_max))
        return np.logspace(log2_N_min, log2_N_max, base=2, num=(log2_N_max-log2_N_min+1), dtype=int)

    Nr_set = log_set(4, Nr_max)
    Ntheta = 16
    L2 = {}
    for i, Nr in enumerate(Nr_set):
        L2_set = test_IC(Nr, Ntheta)
        for label in L2_set:
            if i == 0:
                L2[label] = []
            L2[label].append(L2_set[label])

    fig, ax = plt.subplots(nrows=2)
    linestyles = ['solid', 'solid', 'dashed']

    for label, linestyle in zip(L2, linestyles):
        ax[0].plot(Nr_set, L2[label], label=label, linestyle=linestyle)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log', base=2)
    ax[0].set_xlabel(r'$N_r$')
    ax[0].set_ylabel(r'$L_2(B-B_0)$')
    ax[0].legend(title=r'$N_\theta = '+'{:d}'.format(Ntheta)+r'$')

    Ntheta_set = log_set(4, Ntheta_max)
    Nr = 16
    L2 = {}
    for i, Ntheta in enumerate(Ntheta_set):
        L2_set = test_IC(Nr, Ntheta)
        for label in L2_set:
            if i == 0:
                L2[label] = []
            L2[label].append(L2_set[label])

    for label, linestyle in zip(L2, linestyles):
        ax[1].plot(Ntheta_set, L2[label], label=label, linestyle=linestyle)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log', base=2)
    ax[1].set_xlabel(r'$N_\theta$')
    ax[1].set_ylabel(r'$L_2(B-B_0)$')
    ax[1].legend(title=r'$N_r = '+'{:d}'.format(Nr)+r'$')

    fig.tight_layout()

    filename = 'test_IC_Nr{:d}-{:d}_Nt{:d}-{:d}'.format(Nr_set[0], Nr_set[-1], Ntheta_set[0], Ntheta_set[-1])
    fig.savefig(filename+'.pdf')
    fig.savefig(filename+'.png', dpi=300)

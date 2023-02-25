"""
Dedalus script for shell anelastic convection,
based on Jones et al 2011 convective benchmark case 0.

Usage:
    j2011_hydro.py [options]

Options:
    --Ntheta=<Ntheta>       Latitude coeffs [default: 64]
    --Nr=<Nr>               Radial coeffs  [default: 64]

    --Legendre              Use Legendre polynomials in radius

    --tol=<tol>             Tolerance for opitimization loop [default: 1e-5]
    --eigs=<eigs>           Target number of eigenvalues to search for [default: 20]
    --ncc_cutoff=<ncc>      Amplitude cutoff for NCCs [default: 1e-8]

    --label=<label>         Additional label for run output directory
"""
import sys
import numpy as np
import dedalus.public as de
import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
ncpu = MPI.COMM_WORLD.size

from docopt import docopt
args = docopt(__doc__)

ncc_cutoff = float(args['--ncc_cutoff'])

# parameters
beta = 0.35
Ekman = 2e-3
Prandtl = 1
#Rayleigh = 61621.682
n = 2
Nrho = 3
m_crit = 10

N_eigs = int(float(args['--eigs']))
tol = float(args['--tol'])

Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta

dealias = 3/2
dtype = np.complex128

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

Ro = r_outer = 1/(1-beta)
Ri = r_inner = Ro - 1

zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
zeta_in  = (1 + beta - zeta_out) / beta
c0 = (2*zeta_out - beta - 1) / (1 - beta)
c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype)
if args['--Legendre']:
    basis = de.ShellBasis(coords, alpha=(0,0), shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dtype=dtype)
else:
    basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dtype=dtype)
basis_ncc = basis.meridional_basis
b_inner = basis.S2_basis(radius=r_inner)
b_outer = basis.S2_basis(radius=r_outer)
s2_basis = basis.S2_basis()

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
S = dist.Field(name='S', bases=basis)
τ_p = dist.Field(name='τ_p')
τ_S1 = dist.Field(name='τ_T1', bases=b_outer)
τ_S2 = dist.Field(name='τ_T2', bases=b_inner)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=b_outer)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=b_inner)

grad = lambda A: de.Gradient(A, coords)
div = lambda A: de.Divergence(A)
cross = lambda A, B: de.CrossProduct(A, B)
trans = lambda A: de.TransposeComponents(A)
trace = lambda A: de.Trace(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)

# Substitutions
phi, theta, r = dist.local_grids(basis)
ez = dist.VectorField(coords, bases=basis_ncc, name='ez')
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)

f = 2*ez/Ekman

g = dist.VectorField(coords, bases=basis_ncc, name='g')
g['g'][2] = 1/r**2
rvec = dist.VectorField(coords, bases=basis_ncc, name='rvec')
rvec['g'][2] = r
zeta = dist.Field(bases=basis_ncc, name='zeta')
zeta['g'] = c0 + c1/r

rho0 = (zeta**n).evaluate()
rho0.name='ρ0'
p0 = (zeta**(n+1)).evaluate()
p0.name='p0'
grad_log_rho0 = de.grad(np.log(rho0)).evaluate()
grad_log_rho0.name='grad_ln_ρ0'
grad_log_p0 = de.grad(np.log(p0)).evaluate()
grad_log_p0.name='grad_ln_p0'

er = dist.VectorField(coords, bases=basis_ncc, name='er')
er['g'][2] = 1

lift_basis = basis.derivative_basis(1)
lift = lambda A, n: de.Lift(A, lift_basis, n)

e = grad(u) + trans(grad(u))
viscous_terms = div(e) + e@grad_log_rho0 - 2/3*grad(div(u)) - 2/3*grad_log_rho0*div(u)

trace_e = trace(e)

u_r_inner = radial(u(r=r_inner))
u_r_outer = radial(u(r=r_outer))
u_perp_inner = radial(angular(e(r=r_inner)))
u_perp_outer = radial(angular(e(r=r_outer)))

S0 = dist.Field(bases=basis_ncc, name='S0')
S0['g'] = (zeta_out**(-2) - (c0 + c1/r)**(-2)) / (zeta_out**(-2) - zeta_in**(-2))
grad_S0 = grad(S0).evaluate()

logger.info("NCC expansions:")
for ncc in [grad_log_rho0, grad_log_p0, S0, g]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

Rayleigh = dist.Field(name='Rayleigh')
omega = dist.Field(name='omega')
dt = lambda A: omega*A
# Problem
problem = de.EVP([p, S, u, τ_p, τ_S1, τ_S2, τ_u1, τ_u2], eigenvalue=omega, namespace=locals())
problem.add_equation("div(u) + u@grad_log_rho0 + τ_p + lift(τ_u2,-1)@er = 0")
problem.add_equation("dt(u) + grad(p) - viscous_terms - Rayleigh/Prandtl*S*g - cross(u, f) + lift(τ_u1, -1) + lift(τ_u2, -2) = 0")
problem.add_equation("dt(S) + u@grad_S0 - (lap(S) + grad(S)@grad_log_p0)/Prandtl + lift(τ_S1, -1) + lift(τ_S2, -2) = 0 ")
problem.add_equation("S(r=Ri) = 0")
problem.add_equation("u_r_inner = 0")
problem.add_equation("u_perp_inner = 0")
problem.add_equation("S(r=Ro) = 0")
problem.add_equation("u_r_outer = 0")
problem.add_equation("u_perp_outer = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

Jones_et_al_2011_Rayleigh = 61621.682
Jones_et_al_2011_ω = -1j*101.381 + 0 # adjusts Jones et al 2009 convention of dt = -1j*omega to ours
target = 0 + 0*1j

def compute_eigenvalues(log_Ra_i, target_m=10):
    Rayleigh['g'] = np.exp(log_Ra_i)
    current_Ra = Rayleigh['g'][0,0,0].real
    logger.info('m = {:d}, Ra = {:.6g}'.format(target_m, current_Ra))
    subproblem = solver.subproblems_by_group[(target_m, None, None)]
    solver.solve_sparse(subproblem, N=N_eigs, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    evals /= np.sqrt(current_Ra)
    return(evals)

def peak_growth_rate(*args):
    evals = compute_eigenvalues(*args)
    peak_eval = evals[-1]
    # flip sign so minimize finds maximum
    return np.abs(peak_eval.real)

import scipy.optimize as sciop
log_Ra_i = np.log(6e4) # search in log Ra
bounds = sciop.Bounds(lb=np.log(1e4), ub=np.log(1e5))
result = sciop.minimize(peak_growth_rate, log_Ra_i, bounds=bounds, tol=tol, method='Nelder-Mead')
logger.info(result)

logger.info('optimization complete, solving for critical modes')
evals = compute_eigenvalues(result.x[0])
critical_Ra = np.exp(result.x[0]).real
logger.info("Jones et al 2011 critical Ra: {:.6g}".format(Jones_et_al_2011_Rayleigh))
logger.info("                 critical Ra: {:.6g}".format(critical_Ra))
logger.info("       fractional difference: {:.2g}".format(np.abs(critical_Ra-Jones_et_al_2011_Rayleigh)/np.abs(critical_Ra)))
logger.info("")
Jones_et_al_2011_ω /= np.sqrt(Jones_et_al_2011_Rayleigh)
logger.info("Jones et al 2011  eigenvalue: {:.6g}, {:.6g}i".format(Jones_et_al_2011_ω.real, Jones_et_al_2011_ω.imag))
logger.info("         critical eigenvalue: {:.6g}, {:.6g}i".format(evals[-1].real, evals[-1].imag))
logger.info("       fractional difference: {:.2g}".format(np.abs(evals[-1]-Jones_et_al_2011_ω)/np.abs(evals[-1])))
logger.info("")
logger.info("eigenvalues:\n{:}".format(evals))

target_m=10
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(evals.real, evals.imag, alpha=0.5, zorder=5)
ax.axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.set_title('m = {:d}, '.format(target_m)+r'$N_\theta = '+'{:d}'.format(Ntheta)+r'$, $N_r = '+'{:d}'.format(Nr)+r'$')
ax.set_xlabel(r'$\omega_R$')
ax.set_ylabel(r'$\omega_I$')
ax.scatter(Jones_et_al_2011_ω.real, Jones_et_al_2011_ω.imag, marker='s', label='Jones et. al (2011)', alpha=0.2, color='xkcd:dark red', zorder=2, s=100)
ax.scatter(target.real, target.imag, marker='x', label='target',  color='xkcd:dark green', alpha=0.2, zorder=1)
ax.legend()
fig_filename = 'eigenspectrum'
if args['--Legendre']:
    fig_filename += '_Legendre'
fig.savefig(data_dir+'/'+fig_filename+'.png', dpi=300)

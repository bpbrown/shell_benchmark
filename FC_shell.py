"""
Dedalus script for shell fully compressible convection,
based on Jones et al 2011 convective benchmark case 0.

Usage:
    FC_shell.py [options]

Options:
    --Ntheta=<Ntheta>       Latitude coeffs [default: 64]
    --Nr=<Nr>               Radial coeffs  [default: 64]

    --Mach=<Ma>             Mach number [default: 1e-2]
    --gamma=<gamma>         Ideal gas gamma [default: 5/3]

    --Rayleigh=<Ra>         Rayleigh number [default: 1e6]
    --Ekman=<Ek>            Ekman number [default: 1e-4]

    --Legendre              Use Legendre polynomials in radius

    --jones                 Use the Jones polytrope

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

from fractions import Fraction

ncc_cutoff = float(args['--ncc_cutoff'])

# parameters
beta = 0.35
Ekman = 2e-3
Prandtl = 1
Nrho = nρ = 3
m_crit = 10

Ma = float(args['--Mach'])
Ma2 = Ma*Ma
γ = gamma = float(Fraction(args['--gamma']))
m_ad = 1/(γ-1)
ε = Ma2
m_poly = m_ad - ε

N_eigs = int(float(args['--eigs']))
tol = float(args['--tol'])

Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta

dealias = 3/2
dtype = np.complex128

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Ma{}'.format(args['--Mach'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

if args['--jones']:
    Ro = r_outer = 1/(1-beta)
    Ri = r_inner = Ro - 1
    zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
    zeta_in  = (1 + beta - zeta_out) / beta
    c0 = (2*zeta_out - beta - 1) / (1 - beta)
    c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2
else:
    Ro = r_outer = 1
    Ri = r_inner = beta
    nh = nρ/m_poly
    c0 = -(Ri-Ro*np.exp(-nh))/(Ro-Ri)
    c1 = Ri*Ro/(Ro-Ri)*(1-np.exp(-nh))

logger.info('Ri = {:}, Ro = {:}'.format(Ri, Ro))

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
θ = dist.Field(name='θ', bases=basis)
Υ = dist.Field(name='Υ', bases=basis)
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

T = dist.Field(bases=basis_ncc, name='T')
T['g'] = c0 + c1/r

rho0 = (T**m_poly).evaluate()
rho0.name='ρ0'
log_rho0 = (np.log(rho0)).evaluate()
grad_log_rho0 = grad(log_rho0).evaluate()
grad_log_rho0.name='grad_ln_ρ0'
h0 = (T).evaluate()
h0.name='h0'
grad_h0 = de.grad(h0)
θ0 = (np.log(h0)).evaluate()
θ0.name='θ0'
grad_θ0 = de.grad(θ0).evaluate()
grad_θ0.name='grad_θ0'

S0 = (1/γ*θ0 - (γ-1)/γ*log_rho0).evaluate()
S0.name = 'S0'
grad_S0 = grad(S0).evaluate()

er = dist.VectorField(coords, bases=basis_ncc, name='er')
er['g'][2] = 1

lift_basis = basis #.derivative_basis(1)
lift = lambda A, n: de.Lift(A, lift_basis, n)

e = grad(u) + trans(grad(u))
viscous_terms = div(e) + e@grad_log_rho0 - 2/3*grad(div(u)) - 2/3*grad_log_rho0*div(u)

trace_e = trace(e)

u_r_inner = radial(u(r=r_inner))
u_r_outer = radial(u(r=r_outer))
u_perp_inner = radial(angular(e(r=r_inner)))
u_perp_outer = radial(angular(e(r=r_outer)))

r_g = dist.Field(bases=basis_ncc)
r_g['g'] = r
r_g.name='r'
scale = r_g*T
scale_h = r_g
scale_g = de.Grid(scale).evaluate()
scale_h_g = de.Grid(r_g).evaluate()


logger.info("NCC expansions:")
for ncc in [scale*grad_log_rho0, scale_h*h0, scale_h*grad_h0, scale*grad_S0, scale*grad_θ0, scale]:
#for ncc in [grad_log_rho0, h0, grad_h0, grad_S0, grad_θ0, T]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

g = dist.VectorField(coords, bases=basis_ncc, name='g')
g['g'][2] = 1/r**2

scrC = 1/(gamma-1)/Ma2
#scrC = 1/(gamma-1)*Co2/Ma2
#Co2 = Rayleigh*Ekman**2/Prandtl
logger.info("scrC = {:}".format(scrC))
# Problem  Rayleigh/Prandtl*scrC*(h0*grad(θ) + grad_h0*θ - h0*grad(S)
problem = de.IVP([u, Υ, θ, S, τ_u1, τ_u2, τ_S1, τ_S2], namespace=locals())
problem.add_equation("scale*(dt(Υ) + div(u) + u@grad_log_rho0) + lift(τ_u2,-1)@er = -scale_g*(u@grad(Υ)) ")
problem.add_equation("scale_h*(dt(u) + scrC*Rayleigh*h0*grad(θ) + scrC*Rayleigh*grad_h0*θ - scrC*Rayleigh*h0*grad(S) - viscous_terms) + lift(τ_u1, -1) + lift(τ_u2, -2) = -scale_h_g*(u@e) + scale_h_g*0.5*grad(u@u) + scale_h_g*cross(u, f) - scale_h_g*Rayleigh*scrC*h0_g*(grad_θ0_g*(np.expm1(θ)-θ) + np.expm1(θ)*grad(θ) + np.expm1(θ)*grad(S)) ")
problem.add_equation("scale*(dt(S) + u@grad_S0 - (lap(θ) + 2*grad_θ0@grad(θ))/Prandtl) + lift(τ_S1, -1) + lift(τ_S2, -2) = - scale_g*(u@grad(S)) + scale_g*Prinv_g*(grad(θ)@grad(θ))+ scale_g*Di_zetainv_g*Phi ")
problem.add_equation("θ - (γ-1)*Υ - γ*S = 0")
problem.add_equation("S(r=Ri) = 0")
problem.add_equation("u_r_inner = 0")
problem.add_equation("u_perp_inner = 0")
problem.add_equation("S(r=Ro) = 0")
problem.add_equation("u_r_outer = 0")
problem.add_equation("u_perp_outer = 0")

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

# Solver
solver = problem.build_solver(timestepper, ncc_cutoff=ncc_cutoff)
solver.stop_sim_time = stop_sim_time
# for testing
if args['--niter']:
    solver.stop_iteration = int(float(args['--niter']))

# Initial conditions
# take 𝓁=m spherical harmonic perturbations at 𝓁=[1,19],
# with a radial bump function, and a 𝓁=0 background
amp = 1e-2*Ma2
rnorm = 2*np.pi/(Ro - Ri)
rfunc = (1 - np.cos(rnorm*(r-Ri)))
S['g'] = 0
for 𝓁, amp in zip([1, 19], [1e-3, 1e-2]):
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    S['g'] += amp*norm*rfunc*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁

S.change_scales(1)
θ['g'] = γ*S['g']


# Analysis
eφ = d.VectorField(c, bases=b)
eφ['g'][0] = 1
eθ = d.VectorField(c, bases=b)
eθ['g'][1] = 1
er = d.VectorField(c, bases=b)
er['g'][2] = 1

ur = dot(u, er)
uθ = dot(u, eθ)
uφ = dot(u, eφ)

ρ_cyl = d.Field(bases=b)
ρ_cyl['g'] = r*np.sin(theta)
Ωz = uφ/ρ_cyl # this is not ω_z; misses gradient terms; this is angular differential rotation.

out_cadence = 1e-2

V = basis.volume
azavg = lambda A: de.Average(A, coords.coords[0])
shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V
sphere_integ = lambda A: de.Average(A, coords.S2coordsys)*4*np.pi
L = rho0*cross(rvec,u)
ω = curl(u)*Ekman/2


slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=out_cadence, max_writes=10)
slices.add_task(S(r=Ro*0.98), name='S_r_0.98')
slices.add_task(S(r=(Ri+Ro)/2), name='S_r_mid')
slices.add_task(S(theta=np.pi/2), name='s_eq')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
slices.add_task(azavg(Ωz), name='<Ωz>')
slices.add_task(azavg(S), name='<s>')


profiles = solver.evaluator.add_file_handler(data_dir+'/profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(S(r=(Ri+Ro)/2,theta=np.pi/2), name='S_profile')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=1e-3, max_writes=None)
traces.add_task(0.5*de.integ(rho0*u@u), name='KE')
traces.add_task(np.sqrt(volavg(u@u)), name='Re')
traces.add_task(np.sqrt(volavg(ω@ω)), name='Ro')

traces.add_task(de.integ(L@ex), name='Lx')
traces.add_task(de.integ(L@ey), name='Ly')
traces.add_task(de.integ(L@ez), name='Lz')
traces.add_task(-1/Prandtl*zeta_out**(n+1)*Ro**2*sphere_integ(de.radial(de.grad(S)(r=Ro))), name='Luminosity')

traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_S1)), name='τ_S1')
traces.add_task(shellavg(np.abs(τ_S2)), name='τ_S2')
traces.add_task(shellavg(np.sqrt(τ_u1@τ_u1)), name='τ_u1')
traces.add_task(shellavg(np.sqrt(τ_u2@τ_u2)), name='τ_u2')

# CFL
if args['--max_dt']:
    max_timestep = float(args['--max_dt'])
else:
    max_timestep = Ekman/10

CFL = de.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

report_cadence = 1
# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(u@u), name='Re')
flow.add_property(np.sqrt(ω@ω), name='Ro')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.abs(τ_S1), name='|τ_S1|')
flow.add_property(np.abs(τ_S2), name='|τ_S2|')
flow.add_property(np.sqrt(dot(τ_u1,τ_u1)), name='|τ_u1|')
flow.add_property(np.sqrt(dot(τ_u2,τ_u2)), name='|τ_u2|')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        Δt = CFL.compute_timestep()
        solver.step(Δt)
        if solver.iteration > 0 and solver.iteration % report_cadence == 0:
            max_Re = flow.max('Re')
            avg_Ro = flow.grid_average('Ro')
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_S1|'), flow.max('|τ_S2|'), flow.max('|τ_p|')])

            logger.info('Iteration={:d}, Time={:.4e}, dt={:.1e}, Ro={:.3g}, max(Re)={:.3g}, τ={:.2g}'.format(solver.iteration, solver.sim_time, Δt, avg_Ro, max_Re, max_τ))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

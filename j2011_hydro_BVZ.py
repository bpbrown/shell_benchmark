"""
Dedalus script for shell anelastic convection,
based on Jones et al 2011 convective benchmark case 0.

Usage:
    j2011_hydro.py [options]

Options:

    --niter=<niter>         How many iterations to run for
    --Ntheta=<Ntheta>       Latitude coeffs [default: 128]
    --Nr=<Nr>               Radial coeffs  [default: 128]
    --mesh=<mesh>           Processor mesh for 3-D runs

    --max_dt=<max_dt>       Largest timestep [default: 0.25]
    --end_time=<end_time>   End of simulation, diffusion times [default: 3]

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

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

# parameters
beta = 0.35
Ekman = Ek = 1e-3
Prandtl = Pr = 1
Rayleigh = 351806
m_ad = 2
Nρ = 5
Co2 = Rayleigh*Ekman**2/Prandtl
logger.info('Ek = {}, Ra = {}, Co2 = {}'.format(Ekman, Rayleigh, Co2))


Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta

stop_sim_time = float(args['--end_time'])/Ekman

timestepper = de.SBDF4
dealias = 3/2
dtype = np.float64

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--max_dt']:
    data_dir += '_dt{}'.format(args['--max_dt'])

if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

Ro = r_outer = 1
Ri = r_inner = beta

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
basis_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
b_inner = basis.S2_basis(radius=r_inner)
b_outer = basis.S2_basis(radius=r_outer)
s2_basis = basis.S2_basis()
V = basis.volume

bk1 = basis.clone_with(k=1)
bk2 = basis.clone_with(k=2)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=bk1)
S = dist.Field(name='S', bases=basis)
τ_p = dist.Field(name='τ_p')
τ_S1 = dist.Field(name='τ_S1', bases=b_outer)
τ_S2 = dist.Field(name='τ_S2', bases=b_inner)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=b_outer)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=b_inner)

# Parameters and operators
lift1 = lambda A, n: de.Lift(A, bk1, n)
lift = lambda A, n: de.Lift(A, bk2, n)

ddt = lambda A: de.TimeDerivative(A)

lap = lambda A: de.Laplacian(A, coords)
grad = lambda A: de.Gradient(A, coords)
div = lambda A: de.Divergence(A)
curl = lambda A: de.Curl(A)
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
trans = lambda A: de.TransposeComponents(A)
trace = lambda A: de.Trace(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)
integ = lambda A: de.integ(A)


# Substitutions
phi, theta, r = dist.local_grids(basis)
ex = dist.VectorField(coords, bases=basis, name='ex')
ex['g'][2] = np.sin(theta)*np.cos(phi)
ex['g'][1] = np.cos(theta)*np.cos(phi)
ex['g'][0] = -np.sin(phi)
ey = dist.VectorField(coords, bases=basis, name='ey')
ey['g'][2] = np.sin(theta)*np.sin(phi)
ey['g'][1] = np.cos(theta)*np.sin(phi)
ey['g'][0] = np.cos(phi)
ez = dist.VectorField(coords, bases=bk1, name='ez')
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)

z = dist.Field(name='z', bases=basis)
x = dist.Field(name='x', bases=basis)
y = dist.Field(name='y', bases=basis)
x['g'] = r*np.sin(theta)*np.cos(phi)
y['g'] = r*np.sin(theta)*np.sin(phi)
z['g'] = r*np.cos(theta)


ez_g = de.Grid(ez)

basis_ncc_k2 = basis_ncc.clone_with(k=2)

rvec = dist.VectorField(coords, bases=basis_ncc, name='rvec')
rvec['g'][2] = r

c0 = -(Ri - np.exp(-Nρ/m_ad))/(1-Ri)
c1 = np.exp(-Nρ/m_ad) - c0
T = dist.Field(bases=basis_ncc, name='T')
T['g'] = c0 + c1/r

lnT = (np.log(T)).evaluate()
lnT.name='lnT'
grad_lnT = grad(lnT).evaluate()
grad_lnT.name='grad_lnT'

ρ = (T**m_ad).evaluate()
ρ.name='ρ'
lnρ = (m_ad*lnT).evaluate()
lnρ.name='lnρ'
grad_lnρ = grad(lnρ).evaluate()
grad_lnρ.name='grad_lnρ'
ρT = (ρ*T).evaluate()
ρT.name='ρT'
grad_lnp = (grad_lnT+grad_lnρ).evaluate()
grad_lnp.name='grad_lnp'

er = dist.VectorField(coords, bases=basis.radial_basis, name='er')
er['g'][2] = 1

e = grad(u) + trans(grad(u))

viscous_terms = div(e) + grad_lnρ@e - 2/3*grad(div(u)) - 2/3*grad_lnρ*div(u)

trace_e = trace(e)
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)

m, ell, n = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)
mask = False #(ell==1)*(n==0)

τ_L = dist.VectorField(coords, bases=basis, name='τ_L')
τ_L.valid_modes[2] *= mask
τ_L.valid_modes[0] = False
τ_L.valid_modes[1] = False

b_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)

L_cons_ncc = dist.Field(bases=b_ncc, name='L_cons_ncc')
# suppress aliasing errors in the L_cons_ncc
padded = (1,1,4)
L_cons_ncc.change_scales(padded)
phi_pad, theta_pad, r_pad = dist.local_grids(basis, scales=padded)

R_avg = (Ro+Ri)/2
L_cons_ncc['g'] = (r_pad/R_avg)**3*np.sqrt((r_pad/Ro-1)*(1-r_pad/Ri))
L_cons_ncc.change_scales(1)

logger.info("NCC expansions:")
for ncc in [ρ, T, ρT, L_cons_ncc, L_cons_ncc*ρ, T*grad_lnρ, T*grad_lnT, grad_lnp]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

#Problem
problem = de.IVP([p, u, S, τ_p, τ_u1, τ_u2, τ_S1, τ_S2, τ_L])
problem.add_equation((ρ*ddt(u) + ρ*grad(p) - Co2*ρT*grad(S) - ρ*Ek*viscous_terms + τ_L + lift(τ_u1, -1) + lift(τ_u2, -2),
                      -(ρ*dot(u, e)) + ρ*cross(u, ez_g) ) )
problem.add_equation((L_cons_ncc*ρ*u, 0))
eq = problem.equations[-1]
eq['LHS'].valid_modes[2] *= mask
eq['LHS'].valid_modes[0] = False
eq['LHS'].valid_modes[1] = False

problem.add_equation((T*dot(grad_lnρ, u) + T*div(u) + τ_p + 1/Ek*T*lift(τ_u2,-1)@er, 0))
problem.add_equation((ρT*ddt(S) - Ek/Pr*ρT*(lap(S)+ dot(grad_lnp, grad(S))) + lift(τ_S1, -1) + lift(τ_S2, -2),
                      -(ρT*dot(u, grad(S))) + 1/2*Ek/Co2*Phi))
# Boundary conditions
problem.add_equation((radial(u(r=Ri)), 0))
problem.add_equation((radial(angular(e(r=Ri))), 0))
problem.add_equation((S(r=Ri), 1))
problem.add_equation((radial(u(r=Ro)), 0))
problem.add_equation((radial(angular(e(r=Ro))), 0))
problem.add_equation((S(r=Ro), 0))
problem.add_equation((integ(p), 0))
logger.info("Problem built")

# Solver
solver = problem.build_solver(timestepper, ncc_cutoff=ncc_cutoff)
solver.stop_sim_time = stop_sim_time
# for testing
if args['--niter']:
    solver.stop_iteration = int(float(args['--niter']))

# Initial conditions
# take 𝓁=m spherical harmonic perturbations at 𝓁=[1,19],
# with a radial bump function, and a 𝓁=0 background
rnorm = 2*np.pi/(Ro - Ri)
rfunc = (1 - np.cos(rnorm*(r-Ri)))
S['g'] = 0
for 𝓁, amp in zip([1, 19], [1e-3, 1e-2]):
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    S['g'] += amp*norm*rfunc*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁

# Analysis
out_cadence = 100

shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V

coeffs = solver.evaluator.add_file_handler(data_dir+'/coeffs', sim_dt=20, max_writes = 10)
coeffs.add_task(ρ*u, name='ρu', layout='c')

L = ρ*cross(rvec,u)
ω = curl(u)*Ekman/2

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=5, max_writes=None)
traces.add_task(0.5*integ(ρ*u@u), name='KE')
traces.add_task(0.5*integ(ρ*u@u)/Ekman**2, name='E0')
traces.add_task(np.sqrt(volavg(u@u)), name='Re')
traces.add_task(np.sqrt(volavg(ω@ω)), name='Ro')

traces.add_task(integ(L@ex), name='Lx')
traces.add_task(integ(L@ey), name='Ly')
traces.add_task(integ(L@ez), name='Lz')
traces.add_task(integ(-x*div(L)), name='Λx')
traces.add_task(integ(-y*div(L)), name='Λy')
traces.add_task(integ(-z*div(L)), name='Λz')

traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_S1)), name='τ_S1')
traces.add_task(shellavg(np.abs(τ_S2)), name='τ_S2')
traces.add_task(shellavg(np.sqrt(τ_u1@τ_u1)), name='τ_u1')
traces.add_task(shellavg(np.sqrt(τ_u2@τ_u2)), name='τ_u2')
traces.add_task(shellavg(np.sqrt(τ_L@τ_L)), name='τ_L')

# CFL
max_timestep = float(args['--max_dt'])

CFL = de.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

report_cadence = 10
# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(u@u), name='Re')
flow.add_property(np.sqrt(ω@ω), name='Ro')
flow.add_property(0.5*ρ*u@u/Ekman**2, name='E0')
flow.add_property(L@ez, name='Lz')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.abs(τ_S1), name='|τ_S1|')
flow.add_property(np.abs(τ_S2), name='|τ_S2|')
flow.add_property(np.sqrt(τ_u1@τ_u1), name='|τ_u1|')
flow.add_property(np.sqrt(τ_u2@τ_u2), name='|τ_u2|')
flow.add_property(np.sqrt(τ_L@τ_L), name='|τ_L|')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        Δt = CFL.compute_timestep()
        solver.step(Δt)
        if solver.iteration > 0 and solver.iteration % report_cadence == 0:
            max_Re = flow.max('Re')
            avg_Ro = flow.grid_average('Ro')
            int_Lz = flow.volume_integral('Lz')
            int_E0 = flow.volume_integral('E0')
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_S1|'), flow.max('|τ_S2|'), flow.max('|τ_p|')])
            max_τ_L = flow.max('|τ_L|')
            logger.info('Iteration={:d}, Time={:.2e} ({:.1e}), dt={:.1e}, E0 = {:.3g}, Ro={:.3g}, max(Re)={:.3g}, Lz={:.1e}, τ={:.1e},{:.1e}'.format(solver.iteration, solver.sim_time, solver.sim_time*Ekman, Δt, int_E0, avg_Ro, max_Re, int_Lz, max_τ, max_τ_L))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

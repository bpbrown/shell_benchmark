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

    --max_dt=<max_dt>       Largest timestep
    --end_time=<end_time>   End of simulation, diffusion times [default: 3]

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
Ekman = 1e-3
Prandtl = 1
Rayleigh = 351806
n = 2
Nrho = 5

Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta

stop_sim_time = float(args['--end_time'])

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

Ro = r_outer = 1/(1-beta)
Ri = r_inner = Ro - 1

zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
zeta_in  = (1 + beta - zeta_out) / beta
c0 = (2*zeta_out - beta - 1) / (1 - beta)
c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2
Di = c1*Prandtl/Rayleigh

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
basis_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
b_inner = basis.S2_basis(radius=r_inner)
b_outer = basis.S2_basis(radius=r_outer)
s2_basis = basis.S2_basis()
V = basis.volume

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
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
trans = lambda A: de.TransposeComponents(A)
trace = lambda A: de.Trace(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)

# Substitutions
phi, theta, r = dist.local_grids(basis)
ex = dist.VectorField(coords, bases=basis)
ex['g'][2] = np.sin(theta)*np.cos(phi)
ex['g'][1] = np.cos(theta)*np.cos(phi)
ex['g'][0] = -np.sin(phi)
ey = dist.VectorField(coords, bases=basis)
ey['g'][2] = np.sin(theta)*np.sin(phi)
ey['g'][1] = np.cos(theta)*np.sin(phi)
ey['g'][0] = np.cos(phi)
ez = dist.VectorField(coords, bases=basis)
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)

f = de.Grid(2*ez/Ekman)
omega = de.curl(u)
g = dist.VectorField(coords, bases=basis_ncc)
g['g'][2] = 1/r**2
rvec = dist.VectorField(coords, bases=basis_ncc)
rvec['g'][2] = r
zeta = dist.Field(bases=basis_ncc)
zeta['g'] = c0 + c1/r

rho0 = (zeta**n).evaluate()
rho0.name='ρ0'
p0 = (zeta**(n+1)).evaluate()
p0.name='p0'
grad_log_rho0 = de.grad(np.log(rho0)).evaluate()
grad_log_rho0.name='grad_ln_ρ0'
grad_log_p0 = de.grad(np.log(p0)).evaluate()
grad_log_p0.name='grad_ln_p0'


bk1 = basis.clone_with(k=1)
bk2 = basis.clone_with(k=2)
lift1 = lambda A, n: de.Lift(A, bk1, n)
lift = lambda A, n: de.Lift(A, bk2, n)

e = grad(u) + trans(grad(u))
e.store_last = True
viscous_terms = div(e) + dot(grad_log_rho0, e) - 2/3*grad(div(u)) - 2/3*grad_log_rho0*div(u)

trace_e = trace(e)
trace_e.store_last = True
Phi = trace(dot(e, e)) - 1/3*(trace_e*trace_e)

u_r_inner = radial(u(r=r_inner))
u_r_outer = radial(u(r=r_outer))
u_perp_inner = radial(angular(e(r=r_inner)))
u_perp_outer = radial(angular(e(r=r_outer)))

zetag = de.Grid(zeta)

viscous_heating = Phi

Di_zetainv_g = de.Grid((Di/2)*1/zeta)

# Problem
problem = de.IVP([p, S, u, τ_p, τ_S1, τ_S2, τ_u1, τ_u2], namespace=locals())
problem.add_equation("div(u) + u@grad_log_rho0 + τ_p = 0")
problem.add_equation("dt(u) + grad(p) - viscous_terms - Rayleigh/Prandtl*S*g + lift(τ_u1, -1) + lift(τ_u2, -2) = cross(u, omega + f)")
problem.add_equation("dt(S) - (lap(S) + grad(S)@grad_log_p0)/Prandtl + lift(τ_S1, -1) + lift(τ_S2, -2) = - (u@grad(S)) + Di_zetainv_g*Phi")
problem.add_equation("S(r=Ri) = 1")
problem.add_equation("u_r_inner = 0")
problem.add_equation("u_perp_inner = 0")
problem.add_equation("S(r=Ro) = 0")
problem.add_equation("u_r_outer = 0")
problem.add_equation("u_perp_outer = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Copied from Rayleigh, which itself seems to have been copied from Mark Miesch's implementation in ASH
amp = 0.1
norm = 2*np.pi/(Ro - Ri)
S['g'] = amp*(1 - np.cos(norm*(r-Ri)))*np.sin(19*theta)*np.sin(19*phi)
S['g'] += 0.1*amp*(1 - np.cos(norm*(r-Ri)))*np.sin(theta)*np.sin(phi)
zeta.change_scales(1)
S['g'] += (zeta_out**(-2) - (c0 + c1/r)**(-2)) / (zeta_out**(-2) - zeta_in**(-2))

# Analysis
out_cadence = 1e-2

shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V

snapshots = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=out_cadence, max_writes=10)
snapshots.add_task(S(r=Ro), scales=dealias, name='S_r_outer')
snapshots.add_task(S(r=Ri), scales=dealias, name='S_r_inner')
snapshots.add_task(S(r=(Ri+Ro)/2), scales=dealias, name='S_r_mid')
snapshots.add_task(S(phi=0), scales=dealias, name='S_phi_start')
snapshots.add_task(S(phi=3*np.pi/2), scales=dealias, name='S_phi_end')

profiles = solver.evaluator.add_file_handler(data_dir+'/profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(S(r=(Ri+Ro)/2,theta=np.pi/2), name='S_profile')

sphere_integ = lambda A: de.Average(A, coords.S2coordsys)*4*np.pi
L = rho0*cross(rvec,u)

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=1e-3, max_writes=None)
traces.add_task(0.5*de.integ(rho0*u@u), name='KE')
traces.add_task(de.integ(L@ex), name='Lx')
traces.add_task(de.integ(L@ey), name='Ly')
traces.add_task(de.integ(L@ez), name='Lz')
traces.add_task(-1/Prandtl*zeta_out**(n+1)*Ro**2*sphere_integ(de.radial(de.grad(S)(r=Ro))), name='Luminosity')

traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_S1)), name='τ_S1')
traces.add_task(shellavg(np.abs(τ_S2)), name='τ_S2')
traces.add_task(shellavg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u1')
traces.add_task(shellavg(np.sqrt(dot(τ_u2,τ_u2))), name='τ_u2')
#traces.add_task(volavg(Lz), name='Lz')


# CFL
if args['--max_dt']:
    max_timestep = float(args['--max_dt'])
else:
    max_timestep = Ekman/10

CFL = de.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

report_cadence = 10
# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(u@u), name='Re')
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
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_S1|'), flow.max('|τ_S2|'), flow.max('|τ_p|')])

            logger.info('Iteration={:d}, Time={:.4e}, dt={:.2e}, max(Re)={:.3g}, τ={:.2g}'.format(solver.iteration, solver.sim_time, Δt, max_Re, max_τ))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

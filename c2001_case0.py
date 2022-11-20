"""
Dedalus script for shell boussinesq convection,
based on Christensen et al 2001 convective benchmark case 0.

Usage:
    c2001_case0.py [options]

Options:

    --niter=<niter>         How many iterations to run for
    --Ntheta=<Ntheta>       Latitude coeffs [default: 128]
    --Nr=<Nr>               Radial coeffs  [default: 128]
    --mesh=<mesh>           Processor mesh for 3-D runs

    --max_dt=<max_dt>       Largest timestep

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
Ekman = 1e-3
Prandtl = 1
Rayleigh = 100
Ri = 7/13
Ro = 20/13
Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta
stop_sim_time = 2.5 # already converged to 1e-12 by this point #10

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--max_dt']:
    data_dir += '_dt{}'.format(args['--max_dt'])

if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')



timestepper = de.SBDF4 #de.RK222
dealias = 3/2
dtype = np.float64

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
V = basis.volume

bk1 = basis.clone_with(k=1)
bk2 = basis.clone_with(k=2)

# Fields
p = dist.Field(name='p', bases=bk1)
T = dist.Field(name='T', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
τ_p = dist.Field(name='τ_p')
τ_T1 = dist.Field(name='τ_T1', bases=s2_basis)
τ_T2 = dist.Field(name='τ_T2', bases=s2_basis)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=s2_basis)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=s2_basis)

# Substitutions
phi, theta, r = dist.local_grids(basis)

ez = dist.VectorField(coords, bases=bk1)
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)
f = (2*ez/Ekman).evaluate()

er = dist.VectorField(coords, bases=bk1.radial_basis)
er['g'][2] = 1

rvec = dist.VectorField(coords, bases=bk2.radial_basis)
rvec['g'][2] = r/Ro

lift1 = lambda A, n: de.Lift(A, bk1, n)
lift = lambda A, n: de.Lift(A, bk2, n)

# Problem
problem = de.IVP([p, T, u, τ_p, τ_T1, τ_T2, τ_u1, τ_u2], namespace=locals())
problem.add_equation("div(u) + τ_p + lift1(τ_u2,-1)@er = 0")
problem.add_equation("dt(T) - lap(T)/Prandtl + lift(τ_T1, -1) + lift(τ_T2, -2) = -(u@grad(T))")
problem.add_equation("dt(u) - lap(u) + grad(p)/Ekman - Rayleigh*rvec*T/Ekman + lift(τ_u1, -1) + lift(τ_u2, -2) = cross(u, curl(u) + f)")
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# for testing
if args['--niter']:
    solver.stop_iteration = int(float(args['--niter']))

# Initial conditions
A = 0.1
x = 2*r-Ri-Ro
T['g'] = Ri*Ro/r - Ri + 210*A/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

# Analysis
out_cadence = 1e-2

dot = lambda A, B: de.DotProduct(A, B)
shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V

snapshots = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=1e-2, max_writes=10)
snapshots.add_task(T(r=Ro), scales=dealias, name='T_r_outer')
snapshots.add_task(T(r=Ri), scales=dealias, name='T_r_inner')
snapshots.add_task(T(r=(Ri+Ro)/2), scales=dealias, name='T_r_mid')
snapshots.add_task(T(phi=0), scales=dealias, name='T_phi_start')
snapshots.add_task(T(phi=3*np.pi/2), scales=dealias, name='T_phi_end')

profiles = solver.evaluator.add_file_handler(data_dir+'/profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(T(r=(Ri+Ro)/2,theta=np.pi/2), name='T_profile')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=out_cadence, max_writes=None)
traces.add_task(0.5*volavg(u@u), name='KE')
traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_T1)), name='τ_T1')
traces.add_task(shellavg(np.abs(τ_T2)), name='τ_T2')
traces.add_task(shellavg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u1')
traces.add_task(shellavg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u2')

# CFL
if args['--max_dt']:
    max_timestep = float(args['--max_dt'])
else:
    max_timestep = Ekman/10

CFL = de.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u), name='Re')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.abs(τ_T1), name='|τ_T1|')
flow.add_property(np.abs(τ_T2), name='|τ_T2|')
flow.add_property(np.sqrt(dot(τ_u1,τ_u1)), name='|τ_u1|')
flow.add_property(np.sqrt(dot(τ_u2,τ_u2)), name='|τ_u2|')

report_cadence = 10
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        Δt = CFL.compute_timestep()
        solver.step(Δt)
        if solver.iteration > 0 and solver.iteration % report_cadence == 0:
            max_Re = flow.max('Re')
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_T1|'), flow.max('|τ_T2|'), flow.max('|τ_p|')])

            logger.info('Iteration={:d}, Time={:.4e}, dt={:.2e}, max(Re)={:.3g}, τ={:.2g}'.format(solver.iteration, solver.sim_time, Δt, max_Re, max_τ))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

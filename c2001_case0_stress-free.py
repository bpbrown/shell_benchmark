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
    --end_time=<end_time>   End of simulation, diffusion times [default: 3]

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]

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
Ri = r_inner = 7/13
Ro = r_outer = 20/13
Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta
# at t=2.5, already converged to 1e-12
stop_sim_time = float(args['--end_time'])

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
b_inner = basis.S2_basis(radius=r_inner)
b_outer = basis.S2_basis(radius=r_outer)

s2_basis = basis.S2_basis()
V = basis.volume

bk1 = basis.clone_with(k=1)
bk2 = basis.clone_with(k=2)

# Fields
p = dist.Field(name='p', bases=bk1)
T = dist.Field(name='T', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
τ_p = dist.Field(name='τ_p')
τ_T1 = dist.Field(name='τ_T1', bases=b_outer)
τ_T2 = dist.Field(name='τ_T2', bases=b_inner)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=b_outer)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=b_inner)

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

f = (2*ez/Ekman).evaluate()
f.name = 'f'

er = dist.VectorField(coords, bases=basis.radial_basis, name='er')
er['g'][2] = 1

rvec = dist.VectorField(coords, bases=bk2.radial_basis, name='rvec')
rvec['g'][2] = r/Ro

lift1 = lambda A, n: de.Lift(A, bk1, n)
lift = lambda A, n: de.Lift(A, bk2, n)

radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)
grad = lambda A: de.Gradient(A, coords)
trans = lambda A: de.TransposeComponents(A)
e = grad(u) + trans(grad(u))

m, ell, n = dist.coeff_layout.local_group_arrays(basis.domain, scales=1)
mask = (ell==1)*(n==0)

τ_L = dist.VectorField(coords, bases=basis, name='τ_L')
τ_L.valid_modes[2] *= mask
τ_L.valid_modes[0] = False
τ_L.valid_modes[1] = False

ncc_cutoff = float(args['--ncc_cutoff'])
b_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
#b_ncc = basis.radial_basis

L_cons_ncc = dist.Field(bases=b_ncc, name='L_cons_ncc')
R_avg = (Ro+Ri)/2
L_cons_ncc['g'] = (r/R_avg)**3*np.sqrt((r/Ro-1)*(1-r/Ri))

logger.info("NCC expansions:")
for ncc in [L_cons_ncc, rvec]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

# Problem
problem = de.IVP([p, T, u, τ_p, τ_T1, τ_T2, τ_u1, τ_u2, τ_L], namespace=locals())
problem.add_equation("div(u) + τ_p + lift1(τ_u2,-1)@er = 0")
problem.add_equation("dt(T) - lap(T)/Prandtl + lift(τ_T1, -1) + lift(τ_T2, -2) = -(u@grad(T))")
problem.add_equation("dt(u) - lap(u) + grad(p)/Ekman - Rayleigh*rvec*T/Ekman + τ_L/Ekman + lift(τ_u1, -1) + lift(τ_u2, -2) = cross(u, curl(u) + f)")
problem.add_equation((L_cons_ncc*u, 0))
eq = problem.equations[-1]
eq['LHS'].valid_modes[2] *= mask
eq['LHS'].valid_modes[0] = False
eq['LHS'].valid_modes[1] = False

problem.add_equation("T(r=Ri) = 1")
problem.add_equation("radial(u(r=Ri)) = 0")
problem.add_equation("radial(angular(e(r=Ri))) = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("radial(u(r=Ro)) = 0")
problem.add_equation("radial(angular(e(r=Ro))) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper, ncc_cutoff=ncc_cutoff)
solver.stop_sim_time = stop_sim_time

# for testing
if args['--niter']:
    solver.stop_iteration = int(float(args['--niter']))

# Initial conditions
A = 0.1
χ = 2*r-Ri-Ro
T['g'] = Ri*Ro/r - Ri + 210*A/np.sqrt(17920*np.pi)*(1-3*χ**2+3*χ**4-χ**6)*np.sin(theta)**4*np.cos(4*phi)

# Analysis
out_cadence = 1e-2

dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
div = lambda A: de.Divergence(A, index=0)
curl = lambda A: de.Curl(A)
shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V
integ = lambda A: de.integ(A)

L = cross(rvec, u)*Ekman
ω = curl(u)*Ekman/2
PE = Rayleigh/Ekman*T

snapshots = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=1e-1, max_writes=10)
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
traces.add_task(volavg(PE), name='PE')
traces.add_task(np.sqrt(volavg(u@u)), name='Re')
traces.add_task(np.sqrt(volavg(ω@ω)), name='Ro')
traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_T1)), name='τ_S1')
traces.add_task(shellavg(np.abs(τ_T2)), name='τ_S2')
traces.add_task(shellavg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u1')
traces.add_task(shellavg(np.sqrt(dot(τ_u2,τ_u2))), name='τ_u2')
traces.add_task(shellavg(np.sqrt(dot(τ_L,τ_L))), name='τ_L')
traces.add_task(integ(dot(L,ex)), name='Lx')
traces.add_task(integ(dot(L,ey)), name='Ly')
traces.add_task(integ(dot(L,ez)), name='Lz')
traces.add_task(integ(-x*div(L)), name='Λx')
traces.add_task(integ(-y*div(L)), name='Λy')
traces.add_task(integ(-z*div(L)), name='Λz')

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
flow.add_property(np.sqrt(ω@ω), name='Ro')
flow.add_property(dot(L,ez), name='Lz')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.abs(τ_T1), name='|τ_T1|')
flow.add_property(np.abs(τ_T2), name='|τ_T2|')
flow.add_property(np.sqrt(dot(τ_u1,τ_u1)), name='|τ_u1|')
flow.add_property(np.sqrt(dot(τ_u2,τ_u2)), name='|τ_u2|')
flow.add_property(np.sqrt(dot(τ_L,τ_L)), name='|τ_L|')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        Δt = CFL.compute_timestep()
        solver.step(Δt)
        if solver.iteration > 0 and solver.iteration % report_cadence == 0:
            max_Re = flow.max('Re')
            avg_Ro = flow.grid_average('Ro')
            Lz_int = flow.volume_integral('Lz')
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_T1|'), flow.max('|τ_T2|'), flow.max('|τ_p|')])
            max_τ_L = flow.max('|τ_L|')

            logger.info('Iteration={:d}, Time={:.2e}, dt={:.1e}, Ro={:.2g}, max(Re)={:.2g}, Lz={:.1e}, τ={:.2g},{:.2g}'.format(solver.iteration, solver.sim_time, Δt, avg_Ro, max_Re, Lz_int, max_τ,max_τ_L))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

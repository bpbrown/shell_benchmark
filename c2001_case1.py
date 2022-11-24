"""
Dedalus script for shell boussinesq convection,
based on Christensen et al 2001 convective benchmark case 1.

Usage:
    c2001_case1.py [options]

Options:

    --niter=<niter>         How many iterations to run for
    --Ntheta=<Ntheta>       Latitude coeffs [default: 64]
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
Ekman = 1e-3
Prandtl = 1
MPrandtl = 5
Rayleigh = 100
Ri = r_inner = 7/13
Ro = r_outer = 20/13
Nr = int(args['--Nr'])
Ntheta = int(args['--Ntheta'])
Nphi = 2*Ntheta
stop_sim_time = float(args['--end_time'])
timestepper = de.SBDF2
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


# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
b_inner = basis.S2_basis(radius=r_inner)
b_outer = basis.S2_basis(radius=r_outer)
s2_basis = basis.S2_basis()

bk1 = basis.clone_with(k=1)
bk2 = basis.clone_with(k=2)
V = basis.volume

# Fields
p = dist.Field(name='p', bases=bk1)
T = dist.Field(name='T', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
A = dist.VectorField(coords, name='A', bases=basis)
φ = dist.Field(name='φ', bases=bk1)
τ_p = dist.Field(name='τ_p')
τ_T1 = dist.Field(name='τ_T1', bases=b_outer)
τ_T2 = dist.Field(name='τ_T2', bases=b_inner)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=b_outer)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=b_inner)
τ_A1 = dist.VectorField(coords, name='τ_A1', bases=b_outer)
τ_A2 = dist.VectorField(coords, name='τ_A2', bases=b_inner)
τ_φ = dist.Field(name='τ_φ')

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
ez = dist.VectorField(coords, bases=basis, name='ez')
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)
f = (2*ez/Ekman).evaluate()
f.name='f'
rvec = dist.VectorField(coords, bases=bk2.radial_basis, name='rvec')
rvec['g'][2] = r/Ro
er = dist.VectorField(coords, bases=basis.radial_basis, name='er')
er['g'][2] = 1
B = de.curl(A)

lift1 = lambda A, n: de.Lift(A, bk1, n)
lift = lambda A, n: de.Lift(A, bk2, n)

ell_func_o = lambda ell: ell+1
A_potential_bc_o = de.radial(de.grad(A)(r=Ro)) + de.SphericalEllProduct(A, coords, ell_func_o)(r=Ro)/Ro

ell_func_i = lambda ell: -ell
A_potential_bc_i = de.radial(de.grad(A)(r=Ri)) + de.SphericalEllProduct(A, coords, ell_func_i)(r=Ri)/Ri

# Problem
problem = de.IVP([p, T, u, A, φ, τ_p, τ_T1, τ_T2, τ_u1, τ_u2, τ_A1, τ_A2, τ_φ], namespace=locals())
problem.add_equation("div(u) + τ_p + lift1(τ_u2,-1)@er = 0")
problem.add_equation("dt(T) - lap(T)/Prandtl + lift(τ_T1,-1) + lift(τ_T2,-2) = - (u@grad(T))")
problem.add_equation("dt(u) - lap(u) + grad(p)/Ekman - Rayleigh*rvec*T/Ekman + lift(τ_u1, -1) + lift(τ_u2, -2) = cross(u, curl(u) + f) + cross(B, lap(A))/MPrandtl/Ekman")
problem.add_equation("div(A) + τ_φ + lift1(τ_A2,-1)@er = 0")
problem.add_equation("dt(A) - lap(A)/MPrandtl + grad(φ) + lift(τ_A1, -1) + lift(τ_A2, -2) = cross(u, B)")
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("A_potential_bc_i = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("A_potential_bc_o = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("integ(φ) = 0") # Scalar potential gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
amp = 0.1
x = 2*r-Ri-Ro
T['g'] = Ri*Ro/r - Ri + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

# We want to solve for an initial vector potential A
# with curl(A) = B0. We will do this as a BVP.
B_amp = 3 # 5 is the value in Christensen et al 2001, case 1
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

# Analysis
out_cadence = 1e-2

dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
curl = lambda A: de.Curl(A)
shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V
integ = lambda A: de.integ(A)

L = cross(rvec, u)
ω = curl(u)*Ekman/2

snapshots = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=1e-1, max_writes=10)
snapshots.add_task(T(r=Ro), scales=dealias, name='T_r_outer')
snapshots.add_task(T(r=Ri), scales=dealias, name='T_r_inner')
snapshots.add_task(T(r=(Ri+Ro)/2), scales=dealias, name='T_r_mid')
snapshots.add_task(T(phi=0), scales=dealias, name='T_φ_start')
snapshots.add_task(T(phi=3*np.pi/2), scales=dealias, name='T_φ_end')

profiles = solver.evaluator.add_file_handler(data_dir+'/profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(B(r=(Ri+Ro)/2,theta=np.pi/2), name='B_profile')
profiles.add_task(T(r=(Ri+Ro)/2,theta=np.pi/2), name='T_profile')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=out_cadence, max_writes=None)
traces.add_task(0.5*volavg(u@u), name='KE')
traces.add_task(0.5*volavg(B@B)/Ekman/MPrandtl, name='ME')
traces.add_task(np.sqrt(volavg(u@u)), name='Re')
traces.add_task(np.sqrt(volavg(ω@ω)), name='Ro')
traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(np.abs(τ_φ), name='τ_φ')
traces.add_task(shellavg(np.abs(τ_T1)), name='τ_T1')
traces.add_task(shellavg(np.abs(τ_T2)), name='τ_T2')
traces.add_task(shellavg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u1')
traces.add_task(shellavg(np.sqrt(dot(τ_u2,τ_u2))), name='τ_u2')
traces.add_task(shellavg(np.sqrt(dot(τ_A1,τ_A1))), name='τ_A1')
traces.add_task(shellavg(np.sqrt(dot(τ_A2,τ_A2))), name='τ_A2')
traces.add_task(integ(dot(L,ex)), name='Lx')
traces.add_task(integ(dot(L,ey)), name='Ly')
traces.add_task(integ(dot(L,ez)), name='Lz')

# CFL
if args['--max_dt']:
    max_timestep = float(args['--max_dt'])
else:
    max_timestep = Ekman/10

initial_dt = max_timestep

B_vel = curl(A)/np.sqrt(MPrandtl*Ekman)
CFL = de.CFL(solver, initial_dt=initial_dt, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
#CFL.add_velocity(B_vel)

report_cadence = 100
# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(u@u), name='Re')
flow.add_property(np.sqrt(ω@ω), name='Ro')
flow.add_property(np.abs(τ_p), name='|τ_p|')
flow.add_property(np.abs(τ_φ), name='|τ_φ|')
flow.add_property(np.abs(τ_T1), name='|τ_T1|')
flow.add_property(np.abs(τ_T2), name='|τ_T2|')
flow.add_property(np.sqrt(dot(τ_u1,τ_u1)), name='|τ_u1|')
flow.add_property(np.sqrt(dot(τ_u2,τ_u2)), name='|τ_u2|')
flow.add_property(np.sqrt(dot(τ_A1,τ_A1)), name='|τ_A1|')
flow.add_property(np.sqrt(dot(τ_A2,τ_A2)), name='|τ_A2|')


# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        Δt = CFL.compute_timestep()
        solver.step(Δt)
        if solver.iteration > 0 and solver.iteration % report_cadence == 0:
            max_Re = flow.max('Re')
            avg_Ro = flow.grid_average('Ro')
            max_τ = np.max([flow.max('|τ_u1|'), flow.max('|τ_u2|'), flow.max('|τ_A1|'), flow.max('|τ_A2|'), flow.max('|τ_T1|'), flow.max('|τ_T2|'), flow.max('|τ_p|')])

            logger.info('Iteration={:d}, Time={:.4e}, dt={:.1e}, Ro={:.3g}, max(Re)={:.3g}, τ={:.2g}'.format(solver.iteration, solver.sim_time, Δt, avg_Ro, max_Re, max_τ))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

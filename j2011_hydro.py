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

Ro = 1/(1-beta)
Ri = Ro - 1

zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
zeta_in  = (1 + beta - zeta_out) / beta
c0 = (2*zeta_out - beta - 1) / (1 - beta)
c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2
Di = c1*Prandtl/Rayleigh

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
V = basis.volume

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
S = dist.Field(name='S', bases=basis)
τ_p = dist.Field(name='τ_p')
τ_S1 = dist.Field(name='τ_T1', bases=s2_basis)
τ_S2 = dist.Field(name='τ_T2', bases=s2_basis)
τ_u1 = dist.VectorField(coords, name='τ_u1', bases=s2_basis)
τ_u2 = dist.VectorField(coords, name='τ_u2', bases=s2_basis)

# Substitutions
phi, theta, r = dist.local_grids(basis)
ex = dist.VectorField(coords, bases=basis)
ex['g'][2] = np.cos(theta)
ex['g'][1] = -np.sin(theta)
ey = dist.VectorField(coords, bases=basis)
ey['g'][2] = np.cos(theta)
ey['g'][1] = -np.sin(theta)
ez = dist.VectorField(coords, bases=basis)
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)
f = de.Grid(2*ez/Ekman)
omega = de.curl(u)
g = dist.VectorField(coords, bases=basis.radial_basis)
g['g'][2] = 1/r**2
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r
zeta = dist.Field(bases=basis.radial_basis)
zeta['g'] = c0 + c1/r
rho0 = zeta**n
p0 = zeta**(n+1)
grad_log_rho0 = de.grad(np.log(rho0))
grad_log_p0 = de.grad(np.log(p0))
lift_basis = basis.clone_with(k=1)


lift = lambda A: de.Lift(A, lift_basis, -1)
grad_u = de.grad(u) + rvec*lift(τ_u1) # First-order reduction
grad_S = de.grad(S) + rvec*lift(τ_S1) # First-order reduction
#stress = rho0*(grad_u + de.trans(grad_u))
I = dist.TensorField(coords, name='I', bases=basis.radial_basis)
for i in range(3):
    I['g'][i,i] = 1
stress = grad_u + de.trans(grad_u) - 2/3*de.trace(grad_u)*I
strain = 1/2*(de.grad(u) + de.trans(de.grad(u)))
viscous_heating = 2*(de.trace(strain @ strain) - 1/3*de.div(u)**2)
zetag = de.Grid(zeta)

# Problem
problem = de.IVP([p, S, u, τ_p, τ_S1, τ_S2, τ_u1, τ_u2], namespace=locals())
problem.add_equation("trace(grad_u) + u@grad_log_rho0 + τ_p = 0")
problem.add_equation("dt(u) + grad(p) - (div(stress) + stress@grad_log_rho0) - Rayleigh/Prandtl*S*g + lift(τ_u2) = cross(u, omega + f)")
problem.add_equation("dt(S) - (div(grad_S) + grad_S@grad_log_p0)/Prandtl + lift(τ_S2) = - (u@grad(S)) + Di/zetag*viscous_heating")
problem.add_equation("S(r=Ri) = 1")
problem.add_equation("radial(u(r=Ri)) = 0")
problem.add_equation("angular(radial(strain(r=Ri), index=1)) = 0")
problem.add_equation("S(r=Ro) = 0")
problem.add_equation("radial(u(r=Ro)) = 0")
problem.add_equation("angular(radial(strain(r=Ro), index=1)) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# Copied from Rayleigh, which itself seems to have been copied from Mark (Meish?) implementation in ASH
amp = 0.1
norm = 2*np.pi/(Ro - Ri)
S['g'] = amp*(1 - np.cos(norm*(r-Ri)))*np.sin(19*theta)*np.sin(19*phi)
S['g'] += 0.1*amp*(1 - np.cos(norm*(r-Ri)))*np.sin(theta)*np.sin(phi)
zeta.change_scales((1, 1, 1))
S['g'] += (zeta_out**(-2) - (c0 + c1/r)**(-2)) / (zeta_out**(-2) - zeta_in**(-2))

# Analysis
out_cadence = 1e-2

dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
shellavg = lambda A: de.Average(A, coords.S2coordsys)
volavg = lambda A: de.integ(A)/V

snapshots = solver.evaluator.add_file_handler('slices', sim_dt=out_cadence, max_writes=10)
snapshots.add_task(S(r=Ro), scales=dealias, name='S_r_outer')
snapshots.add_task(S(r=Ri), scales=dealias, name='S_r_inner')
snapshots.add_task(S(r=(Ri+Ro)/2), scales=dealias, name='S_r_mid')
snapshots.add_task(S(phi=0), scales=dealias, name='S_phi_start')
snapshots.add_task(S(phi=3*np.pi/2), scales=dealias, name='S_phi_end')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(S(r=(Ri+Ro)/2,theta=np.pi/2), name='S_profile')

traces = solver.evaluator.add_file_handler('traces', sim_dt=1e-3, max_writes=None)
traces.add_task(0.5*de.integ(rho0*u@u), name='KE')
traces.add_task(de.integ(rho0*de.cross(rvec,u)@ex), name='Lx')
traces.add_task(de.integ(rho0*de.cross(rvec,u)@ey), name='Ly')
traces.add_task(de.integ(rho0*de.cross(rvec,u)@ez), name='Lz')
sphere_integ = lambda A: de.Average(A, coords.S2coordsys)*4*np.pi
traces.add_task(-1/Prandtl*zeta_out**(n+1)*Ro**2*sphere_integ(de.radial(de.grad(S)(r=Ro))), name='Luminosity')
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=out_cadence, max_writes=None)
traces.add_task(np.abs(τ_p), name='τ_p')
traces.add_task(shellavg(np.abs(τ_T1)), name='τ_T1')
traces.add_task(shellavg(np.abs(τ_T2)), name='τ_T2')
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

# Flow properties
flow = de.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(u@u), name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 1 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

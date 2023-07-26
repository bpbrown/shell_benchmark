"""
Dedalus script for conducting Boussinesq convection
Case 0 from 'A numerical dynamo benchmark', Christensen et. al.)
Checkpoints are saved which can be restarted from by using the --restart option

Usage:
    case0.py [options]

Options:

    --L_max=<L_max>         Max spherical harmonic [default: 43]
    --N_max=<N_max>         Max radial polynomial  [default: 48]
    --mesh=<mesh>           Processor mesh for 3-D runs
    --restart               Whether to restart or not
"""

import numpy as np
import dedalus.public as d3

from mpi4py import MPI

import logging

from docopt import docopt
args = docopt(__doc__)

restart_file=False
comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

logger = logging.getLogger(__name__)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    # Try to create balanced mesh                                                                                                                                                                                  
    # Choose mesh whose factors are most similar in size                                                                                                                                                           
    factors = [[ncpu//i,i] for i in range(1,int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
    score = np.array([f[1]/f[0] for f in factors])
    mesh = factors[np.argmax(score)]

logger.info("running on processor mesh={}".format(mesh))
Lmax = int(args['--L_max'])
Nmax = int(args['--N_max'])
restart = args['--restart']
# Parameters
Ekman = 1e-3
Prandtl = 1
Rayleigh = 100

r_inner = 7/13
r_outer = 20/13

# Domain
radii = (r_inner,r_outer)
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, mesh=mesh,dtype=np.float64)
b = d3.ShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii,dealias=3/2, dtype=np.float64)
s2_basis = b.S2_basis()

phi, theta, r = d.local_grids(b)

# Fields for IVP
u = d.VectorField(c,name='u',bases=b)
A = d.VectorField(c,name='A',bases=b)
p = d.Field(name='p',bases=b)
T = d.Field(name='T',bases=b)

tau_u1 = d.VectorField(c,name='tau_u1',bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2',bases=s2_basis)

tau_p = d.Field(name='tau_p')

tau_T1 = d.Field(name='tau_T1',bases=s2_basis)
tau_T2 = d.Field(name='tau_T2',bases=s2_basis)

ez = d.VectorField(c,name='ez',bases=b)

ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec =  d.VectorField(c,name='r_vec',bases=b.radial_basis)
r_vec['g'][2] = r/r_outer

rvec = d.VectorField(c, name='rvec', bases=b.radial_basis)
rvec['g'][2] = r

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction

if(restart):
    write, initial_timestep = solver.load_state('case0_checkpoints/case0_checkpoints_s1.h5')
else:
    # initial condition
    amp = 0.1
    x = 2*r-r_inner-r_outer
    T['g'] = r_inner*r_outer/r - r_inner + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

problem = d3.IVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2, tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("Ekman*dt(u) - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) = cross(u, Ekman*curl(u) + 2*ez) + Rayleigh*r_vec*T")
problem.add_equation("dt(T) - div(grad_T)/Prandtl + lift(tau_T2,-1) = - dot(u,grad(T))")

# Gauge conditions
problem.add_equation("integ(p)=0")

# Boundary conditions
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 1")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

property_cadence = 100
# Solver
solver = problem.build_solver(d3.SBDF2)
logger.info("Problem built")

solver.stop_sim_time = 15

vol = 4*np.pi/3*(r_outer**3-r_inner**3)
flow = d3.GlobalFlowProperty(solver, cadence=property_cadence)
flow.add_property(d3.dot(u,u)/2., name='KE_density')

max_timestep = 1e-4
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.8, threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

good_solution = True

timeseries = solver.evaluator.add_file_handler('case0_timeseries', sim_dt = 1e-3, mode='append')
timeseries.add_task(0.5*d3.integ(d3.dot(u,u))/vol,name='KE')

snapshots = solver.evaluator.add_file_handler('case0_snapshots', sim_dt = 2.5, mode='append')
snapshots.add_task(u,name='u')
snapshots.add_task(T,name='T')

checkpoints = solver.evaluator.add_file_handler('case0_checkpoints', sim_dt = 0.5, mode='overwrite')
checkpoints.add_tasks(solver.state)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed and good_solution:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration-1) % property_cadence == 0:
            KE = flow.volume_integral('KE_density')/vol

            good_solution = np.isfinite(KE)
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%f' %(solver.iteration, solver.sim_time, timestep, KE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

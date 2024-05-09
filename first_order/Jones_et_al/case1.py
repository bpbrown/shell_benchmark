"""
Dedalus script for shell anelastic convection,
Case 1 from Jones et al (2011), "Anelastic convection-driven dynamo benchmarks"
Checkpoints are saved which can be restarted from by using the --restart option

Usage:
    case1.py [options]

Options:

    --L_max=<L_max>         Max spherical harmonic [default: 127]
    --N_max=<N_max>         Max radial polynomial  [default: 63]
    --mesh=<mesh>           Processor mesh for 3-D runs
    --restart               Whether to restart or not
"""
import numpy as np
import dedalus.public as d3

from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import time

from docopt import docopt
args = docopt(__doc__)

import logging

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
beta = 0.35

Ek = 1e-3
Pr = 1
Ra = 351806
n = 2
Nrho = 5

r_outer = 1/(1-beta)
r_inner = r_outer - 1
radii = (r_inner,r_outer)

timestepper = d3.SBDF2

# Coordinates and domain
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), mesh=mesh,dtype=np.float64)
b = d3.ShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii, dealias=3/2, dtype=np.float64)
radial_basis = d3.ShellBasis(c, (1,1,Nmax+1), radii=radii, dealias=3/2, dtype=np.float64)
s2_basis = b.S2_basis()

phi, theta, r = d.local_grids(b)

# problem variables
u   = d.VectorField(c,name='u',bases=b)
p   = d.Field(name='p',bases=b)
S   = d.Field(name='S',bases=b)

tau_u1 = d.VectorField(c,name='tau_u1',bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2',bases=s2_basis)

tau_S1 = d.Field(name='tau_S1',bases=s2_basis)
tau_S2 = d.Field(name='tau_S2',bases=s2_basis)
tau_p = d.Field(name='tau_p')

# Extra fields used to define the problem
ez = d.VectorField(c,name='ez',bases=b.meridional_basis)

ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

g = d.VectorField(c, name='g', bases=radial_basis)
g['g'][2] = 1/r**2

# Polytropic gas terms
zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
zeta_in  = (1 + beta - zeta_out) / beta
c0 = (2*zeta_out - beta - 1) / (1 - beta)
c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2

Di = c1*Pr/(Ra)

assert(np.allclose(c0 + c1/r_inner, zeta_in)) # zeta_in is correct
assert(np.allclose(c0 + c1/r_outer, zeta_out)) # zeta_out is correct
assert(np.allclose( (c0 + c1/r_inner)**n/(c0 + c1/r_outer)**n, np.exp(Nrho))) # Nrho is correct
zetag = c0 + c1/r

zeta = d.Field(name='zeta', bases=radial_basis)
zetainv = d.Field(name='zetainv', bases=radial_basis)
zeta['g'] = zetag
zetainv['g'] = 1/zetag

rho0    = d.Field(name='rho0', bases=radial_basis)
rho0inv = d.Field(name='rho0inv', bases=radial_basis)
logrho0 = d.Field(name='logrho0', bases=radial_basis)

rho0['g'] = zetag**n
rho0inv['g'] = 1/rho0['g']
rho = zetag**n
logrho0['g'] = n*np.log(zetag)
gradlogrho0 = d3.grad(logrho0).evaluate()

p0    = d.Field(name='p0', bases=radial_basis)
p0inv = d.Field(name='p0inv', bases=radial_basis)
logp0 = d.Field(name='logp0', bases=radial_basis)

p0['g'] = zetag**(n+1)
p0inv['g'] = 1/p0['g']
logp0['g'] = (n+1)*np.log(zetag)
gradlogp0 = d3.grad(logp0).evaluate()

amp = 1e-3
norm = 2*np.pi/(r_outer - r_inner)

rvec = d.VectorField(c, name='rvec', bases=b.radial_basis)
rvec['g'][2] = r

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_S = d3.grad(S) + rvec*lift(tau_S1,-1) # First-order reduction

# Viscous and heating terms
e = d3.grad(u) + d3.trans(d3.grad(u))
e_fo = grad_u + d3.trans(grad_u)

viscous_terms = d3.div(e_fo) + d3.dot(gradlogrho0, e_fo) - 2/3*d3.grad(d3.trace(grad_u)) - 2/3*gradlogrho0*d3.trace(grad_u)

trace_e = d3.trace(e)

Phi = d3.trace(d3.dot(e, e)) - 1/3*(trace_e*trace_e)

# Problem
problem = d3.IVP([u, p, tau_u1, tau_u2, tau_p, S, tau_S1, tau_S2], namespace=locals())
problem.add_equation("dt(u) + grad(p) - Ra/Pr*g*S - viscous_terms + lift(tau_u2,-1) = - dot(u, e) - 2/Ek*cross(ez, u)")
problem.add_equation("dot(gradlogrho0, u) + trace(grad_u) + tau_p = 0")
problem.add_equation("dt(S) - 1/Pr*div(grad_S) - 1/Pr*dot(gradlogp0, grad_S) + lift(tau_S2,-1) = - dot(u, grad(S)) + (Di/2)*(zetainv*(Phi))")
problem.add_equation("integ(p)=0")

problem.add_equation("radial(u(r=r_inner)) = 0")
problem.add_equation("angular(radial(e(r=r_inner),0),0) = 0")
problem.add_equation("radial(u(r=r_outer)) = 0")
problem.add_equation("angular(radial(e(r=r_outer),0),0) = 0")

problem.add_equation("S(r=r_inner) = 1")
problem.add_equation("S(r=r_outer) = 0")

solver = problem.build_solver(timestepper)
logger.info("Problem built")

if(restart):
    write, initial_timestep = solver.load_state('case1_checkpoints/case1_checkpoints_s1.h5')
else:
    # Target m=19 mode. 
    initial_timestep = 1e-5
    amp = 1e-3
    norm = 2*np.pi/(r_outer - r_inner)
    S['g'] = amp*(1 - np.cos(norm*(r-r_inner)))*np.sin(19*theta)*np.sin(19*phi)
    S['g'] += 0.1*amp*(1 - np.cos(norm*(r-r_inner)))*np.sin(theta)*np.sin(phi)

    S['g'] += (zeta_out**(-2) - zetag**(-2)) / (zeta_out**(-2) - zeta_in**(-2))

timeseries = solver.evaluator.add_file_handler('case1_timeseries', sim_dt = 1e-5, mode='append')
timeseries.add_task(0.5*d3.integ(rho0*d3.dot(u,u)),name='KE')
timeseries.add_task(0.5*d3.integ(d3.dot(B,B))*Pm/Ek, name='ME')

snapshots = solver.evaluator.add_file_handler('case1_snapshots', sim_dt = 0.35, mode='append')
snapshots.add_task(u,name='u')
snapshots.add_task(S,name='S')

checkpoints = solver.evaluator.add_file_handler('case1_checkpoints', sim_dt = 0.01, mode='overwrite')
checkpoints.add_tasks(solver.state)

property_cadence = 1000
flow = d3.GlobalFlowProperty(solver, cadence=property_cadence)
flow.add_property(0.5*rho0*d3.dot(u,u), name='KE_density')

max_timestep = 1e-5

CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.4, threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

solver.stop_sim_time = 1.4
good_solution = True
try:
    logger.info('Starting main loop')
    while solver.proceed and good_solution:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration-1) % property_cadence == 0:
            KE = flow.volume_integral('KE_density')
            good_solution = np.isfinite(KE)
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%g' %(solver.iteration, solver.sim_time, timestep, KE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

"""
Dedalus script for conducting Boussinesq convection
Case 1 from 'A numerical dynamo benchmark', Christensen et. al.)
Checkpoints are saved which can be restarted from by using the --restart option

Usage:
    case1.py [options]

Options:

    --L_max=<L_max>         Max spherical harmonic [default: 73]
    --N_max=<N_max>         Max radial polynomial  [default: 78]
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
Pm = 5

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

tau_V = d.Field(name='tau_V')

phi_shell = d.Field(name='phi_shell',bases=b)
T         = d.Field(name='T',bases=b)

tau_u1 = d.VectorField(c,name='tau_u1',bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2',bases=s2_basis)
tau_A1 = d.VectorField(c,name='tau_A1',bases=s2_basis)
tau_A2 = d.VectorField(c,name='tau_A2',bases=s2_basis)
tau_p = d.Field(name='tau_p')
tau_phi = d.Field(name='tau_phi')

tau_T1 = d.Field(name='tau_T1',bases=s2_basis)
tau_T2 = d.Field(name='tau_T2',bases=s2_basis)

ez = d.VectorField(c,name='ez',bases=b)

ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec =  d.VectorField(c,name='r_vec',bases=b.radial_basis)
r_vec['g'][2] = r/r_outer

# Fields for initial BVP
B_init = d.VectorField(c,name='B_init',bases=b)
V = d.Field(name='V',bases=b)

# Substitutions
J = d3.curl(d3.curl(A))
B = d3.curl(A)
J_init = d3.curl(B_init)

# Boundary condition terms and first order reduction
ell_func_inner = lambda ell: ell
ellmult = lambda A: d3.SphericalEllProduct(A, c, ell_func_inner)

ell_func_outer = lambda ell: ell+1
ellp1mult = lambda A: d3.SphericalEllProduct(A, c, ell_func_outer)

rvec = d.VectorField(c, name='er', bases=b.radial_basis)

rvec['g'][2] = r

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction
grad_A = d3.grad(A) + rvec*lift(tau_A1,-1) # First-order reduction

if(restart):
    write, initial_timestep = solver.load_state('case1_checkpoints/case1_checkpoints_s1.h5')
else:
    # initial condition
    amp = 0.1
    x = 2*r-r_inner-r_outer
    T['g'] = r_inner*r_outer/r - r_inner + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

    # Solve LBVP to calculate vector potential from B

    B_init['g'][2] = 5/8*(8*r_outer - 6*r -2*r_inner**4/r**3)*np.cos(theta)
    B_init['g'][1] = 5/8*(9*r - 8*r_outer - r_inner**4/r**3)*np.sin(theta)
    B_init['g'][0] = 5*np.sin(np.pi*(r-r_inner))*np.sin(2*theta)

    BVP = d3.LBVP([A, V, tau_V,tau_A1,tau_A2], namespace=locals())
    BVP.add_equation("-lap(A) + grad(V) + lift(tau_A1,-1) + lift(tau_A2,-2) = J_init")
    BVP.add_equation("div(A) + tau_V = 0")
    BVP.add_equation("integ(V) = 0")
    BVP.add_equation("radial(grad(A)(r=r_inner))  - ellmult(A)(r=r_inner)/r_inner = 0")
    BVP.add_equation("radial(grad(A)(r=r_outer)) + ellp1mult(A)(r=r_outer)/r_outer = 0")
    solver_BVP = BVP.build_solver()

    solver_BVP.solve()
    logger.info("solved BVP")

problem = d3.IVP([p, u, T,A,phi_shell, tau_u1,tau_u2,tau_T1,tau_T2,tau_A1,tau_A2,tau_p,tau_phi], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("Ekman*dt(u) - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) = cross(u, Ekman*curl(u) + 2*ez) + Rayleigh*r_vec*T + 1./Pm*cross(J,B)")
problem.add_equation("dt(T) - div(grad_T)/Prandtl + lift(tau_T2,-1) = - dot(u,grad(T))")

problem.add_equation("dt(A) - 1./Pm*div(grad_A) + grad(phi_shell) + lift(tau_A2,-1) = cross(u, B)")
problem.add_equation("trace(grad_A) + tau_phi= 0")

# Gauge conditions
problem.add_equation("integ(p)=0")
problem.add_equation("integ(phi_shell)=0")

# Boundary conditions
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 1")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

problem.add_equation("radial(grad(A)(r=r_inner))  - ellmult(A)(r=r_inner)/r_inner = 0")
problem.add_equation("radial(grad(A)(r=r_outer)) + ellp1mult(A)(r=r_outer)/r_outer = 0")

property_cadence = 100
# Solver
solver = problem.build_solver(d3.SBDF2)
logger.info("Problem built")

solver.stop_sim_time = 15

vol = 4*np.pi/3*(r_outer**3-r_inner**3)
flow = d3.GlobalFlowProperty(solver, cadence=property_cadence)
flow.add_property(d3.dot(u,u)/2., name='KE_density')
flow.add_property(0.5*d3.dot(B,B)/(Ekman*Pm), name='ME_density')

max_timestep = 1e-4
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.8, threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

good_solution = True

timeseries = solver.evaluator.add_file_handler('case1_timeseries', sim_dt = 1e-3, mode='append')
timeseries.add_task(0.5*d3.integ(d3.dot(B,B))/(Ekman*Pm)/vol,name='ME')
timeseries.add_task(0.5*d3.integ(d3.dot(u,u))/vol,name='KE')

snapshots = solver.evaluator.add_file_handler('case1_snapshots', sim_dt = 2.5, mode='append')
snapshots.add_task(u,name='u')
snapshots.add_task(B,name='B')
snapshots.add_task(T,name='T')

checkpoints = solver.evaluator.add_file_handler('case1_checkpoints', sim_dt = 0.5, mode='overwrite')
checkpoints.add_tasks(solver.state)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed and good_solution:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration-1) % property_cadence == 0:
            KE = flow.volume_integral('KE_density')/vol
            ME = flow.volume_integral('ME_density')/vol

            good_solution = np.isfinite(KE) and np.isfinite(ME)
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%f, ME=%f' %(solver.iteration, solver.sim_time, timestep, KE, ME))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

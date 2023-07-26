"""
Dedalus script for conducting Boussinesq convection
Case 2 from 'A numerical dynamo benchmark', Christensen et. al.)
Checkpoints are saved which can be restarted from by using the --restart option

Usage:
    case2.py [options]

Options:

    --L_max=<L_max>         Max spherical harmonic [default: 73]
    --N_max=<N_max>         Max radial polynomial  [default: 78]
    --mesh=<mesh>           Processor mesh for 3-D runs
    --restart               Whether to restart or not
"""

import numpy as np
import dedalus.public as d3

from mpi4py import MPI

from docopt import docopt
args = docopt(__doc__)

import logging

from dedalus.extras.flow_tools import GlobalArrayReducer
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

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
Rayleigh = 110
Pm = 5

r_inner = 7/13
r_outer = 20/13

# Domain
radii = (r_inner,r_outer)
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, mesh=mesh,dtype=np.float64)
b = d3.ShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii,dealias=3/2, dtype=np.float64)
s2_basis = b.S2_basis()

# Inner core basis
ball = d3.BallBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radius=radii[0],dealias=3/2, dtype=np.float64)
sphere = ball.surface
phi_in, theta_in, r_in = d.local_grids(ball)

phi, theta, r = d.local_grids(b)

# Fields for IVP
u = d.VectorField(c,name='u',bases=b)
A = d.VectorField(c,name='A',bases=b)

p = d.Field(name='p',bases=b)

tau_V = d.Field(name='tau_V')

phi_shell = d.Field(name='phi_shell',bases=b)
T = d.Field(name='T',bases=b)

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

B_init_in = d.VectorField(c,name='B_init_in',bases=ball)


# Unit vectors
e_r = d.VectorField(c, name='e_r')
e_r['g'][2] = 1

e_r2 = d.VectorField(c, name='e_r2')
e_r2['g'][2] = 1

e_phi = d.VectorField(c, name='e_phi')
e_phi['g'][0] = 1

e_theta = d.VectorField(c, name='e_theta')
e_theta['g'][1] = 1

# Terms for inner core
A_in = d.VectorField(c,name='A_in',bases=ball)
tau_A_in = d.VectorField(c,name='tau_A_in ',bases=sphere)

lift_in = lambda A, n: d3.Lift(A, ball, n)
phi_in = d.Field(name='phi_in',bases=ball)
u_in = d.VectorField(c,name='u_in',bases=ball)
tau_phi_in = d.Field(name='tau_phi_in')

# Substitutions
B = d3.curl(A)
J = d3.curl(B)

B_in = d3.curl(A_in)
J_in  = d3.curl(B_in)

# Terms for rotation

om = d.Field(name='om')
omRHS = d.Field(name='omRHS')

shell_integral = lambda A: d3.Average(A, c.S2coordsys)*4*np.pi*r_inner**2

sinT = d.Field(name='sinT',bases = b)
sinT['g'] = np.sin(theta)

cyl_rad = d.Field(name='cyl_rad',bases = b)
cyl_rad['g'] = r*np.sin(theta)

torque_mag = r_inner*shell_integral((d3.dot(e_r,B)*d3.dot(e_phi,B)*sinT)(r=r_inner))
torque_vis = Ekman*Pm*shell_integral((cyl_rad**2*d3.dot(e_r,d3.grad(d3.dot(e_phi,u)/cyl_rad)))(r=r_inner))
I = 8/15*np.pi*r_inner**5

u_rotate = d.VectorField(c,name='uin',bases=ball)
u_rotate['g'][0] = r_inner*np.sin(theta)
u_rotate['g'][1] = 0
u_rotate['g'][2] = 0

# Boundary condition terms and first order reduction
ell_func_inner = lambda ell: ell
ellmult = lambda A: d3.SphericalEllProduct(A, c, ell_func_inner)

ell_func_outer = lambda ell: ell+1
ellp1mult = lambda A: d3.SphericalEllProduct(A, c, ell_func_outer)

rvec = d.VectorField(c, name='rvec', bases=b.radial_basis)

rvec['g'][2] = r

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction
grad_A = d3.grad(A) + rvec*lift(tau_A1,-1) # First-order reduction

# initial condition
if(restart):
    write, initial_timestep = solver.load_state('case2_checkpoints/case2_checkpoints_s1.h5')
else:
    amp = 0.1
    x = 2*r-r_inner-r_outer
    T['g'] = r_inner*r_outer/r - r_inner + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

    # Specify A and check its curl is B
    # Not divergence free but will be after one timestep

    B_init['g'][2] = 5*(4*r_outer-3*r)/(3+r_outer)*np.cos(theta)
    B_init['g'][1] = 5*(9*r-8*r_outer)/(6+2*r_outer)*np.sin(theta)
    B_init['g'][0] = 5*np.sin(np.pi*r/r_outer)*np.sin(2*theta)

    A['g'][2] = 5/2*r*np.sin(r*np.pi/r_outer)*np.cos(2*theta)
    A['g'][0] = 5*r*(-3*r+4*r_outer)*np.sin(theta)/(2*(r_outer+3))

    A_in['g'][2] = 5/2*r*np.sin(r*np.pi/r_outer)*np.cos(2*theta)
    A_in['g'][0] = 5*r*(-3*r+4*r_outer)*np.sin(theta)/(2*(r_outer+3))

    vol = 4*np.pi/3*(r_outer**3-r_inner**3)
    error = reducer.global_max(d3.integ(d3.dot(d3.curl(A)-B_init,d3.curl(A)-B_init)).evaluate()['g'])
    logger.info('IC error = %f' % (error/vol))

# General function

def calc_rhs():
    rhs = (torque_vis + torque_mag).evaluate()['g']
    return rhs

om_RHS = d3.GeneralFunction(d, omRHS.domain, (), np.float64, 'g', calc_rhs) 


problem = d3.IVP([p, u, T,A,phi_shell, tau_u1,tau_u2,tau_T1,tau_T2,tau_A1,tau_A2,tau_p,A_in,tau_A_in,phi_in,tau_phi,tau_phi_in,om], namespace=locals())
# Equations in the shell
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("Ekman*dt(u) - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) = cross(u, Ekman*curl(u) + 2*ez) + Rayleigh*r_vec*T + 1./Pm*cross(J,B)")
problem.add_equation("dt(T) - div(grad_T)/Prandtl + lift(tau_T2,-1) = - dot(u,grad(T))")

problem.add_equation("dt(A) - 1./Pm*div(grad_A) + grad(phi_shell) + lift(tau_A2,-1) = cross(u, B)")
problem.add_equation("trace(grad_A) + tau_phi = 0")

# Gauge conditions
problem.add_equation("integ(p)=0")
problem.add_equation("integ(phi_shell)=0")

# Boundary conditions
problem.add_equation("u(r=r_inner) = om*u_rotate(r=r_inner)")
problem.add_equation("T(r=r_inner) = 1")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

problem.add_equation("radial(grad(A)(r=r_outer)) + ellp1mult(A)(r=r_outer)/r_outer = 0") # Potential BC

problem.add_equation("phi_in(r=r_inner) = 0") # Continuity of electric field
problem.add_equation("phi_shell(r=r_inner) = 0") # Continuity of electric field
problem.add_equation("angular(A_in(r=r_inner)-A(r=r_inner)) = 0") # Continuity of radial magnetic field
problem.add_equation("angular(curl(A)(r=r_inner)-curl(A_in)(r=r_inner)) = 0") # Continuity of angular magnetic field

# Inner core
problem.add_equation("dt(A_in) - 1./Pm*lap(A_in) + lift_in(tau_A_in,-1) + grad(phi_in) =  cross(om*u_rotate, B_in)")
problem.add_equation("div(A_in) +  tau_phi_in = 0")
problem.add_equation("integ(phi_in) = 0")

# Spinning of inner core due to torques
problem.add_equation("Ekman*Pm*I*dt(om) = om_RHS")

property_cadence = 100
# Solver
solver = problem.build_solver(d3.SBDF2)
logger.info("Problem built")

solver.stop_sim_time = 15

vol = b.volume
flow = d3.GlobalFlowProperty(solver, cadence=property_cadence)
flow.add_property(d3.dot(u,u)/2., name='KE_density')
flow.add_property(0.5*d3.dot(B,B)/(Ekman*Pm), name='ME_density')
flow.add_property(torque_vis, name='V_torque')
flow.add_property(torque_mag, name='M_torque')
flow.add_property(om, name='spin')

good_solution = True

timeseries = solver.evaluator.add_file_handler('case2_timeseries', sim_dt = 1e-3, mode='append')
timeseries.add_task(0.5*d3.integ(d3.dot(B,B))/(Ekman*Pm)/vol,name='ME')
timeseries.add_task(0.5*d3.integ(d3.dot(u,u))/vol,name='KE')
timeseries.add_task(torque_vis,name='torque_vis')
timeseries.add_task(torque_mag,name='torque_mag')
timeseries.add_task(torque_mag/Ekman/Pm,name='Gamma_L')
timeseries.add_task(om,name='om')

snapshots = solver.evaluator.add_file_handler('case2_snapshots', sim_dt = 2.5, mode='append')
snapshots.add_task(u,name='u')
snapshots.add_task(B,name='B')
snapshots.add_task(B_in,name='B_in')
snapshots.add_task(T,name='T')

checkpoints = solver.evaluator.add_file_handler('case2_checkpoints', sim_dt = 0.5, mode='overwrite')
checkpoints.add_tasks(solver.state)

CFL = d3.CFL(solver, initial_dt=1e-4, cadence=10, safety=0.8, threshold=0.1, max_change=1.5, min_change=0.5, max_dt=1e-4)
CFL.add_velocity(u)
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed and good_solution:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration - 1) % property_cadence == 0:
            KE = flow.volume_integral('KE_density')/vol
            ME = flow.volume_integral('ME_density')/vol
            T_V = flow.max('V_torque')
            T_M = flow.max('M_torque')
            spin = flow.max('spin')

            good_solution = np.isfinite(KE) and np.isfinite(ME)
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%f, ME=%f, TV = %f, TM = %f, spin = %f, Gamma_L = %f' %(solver.iteration, solver.sim_time, timestep, KE, ME,T_V,T_M,spin,T_M/Ekman/Pm))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

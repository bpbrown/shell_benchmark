"""
Dedalus script for shell boussinesq convection,
based on Christensen et al 2001 convective benchmark case 0.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# parameters
Ekman = 1e-3
Prandtl = 1
Rayleigh = 100
Ri = 7/13
Ro = 20/13
Nr = 128
Lmax = 127 #63
Ntheta = Lmax + 1
Nphi = 2*Ntheta
stop_sim_time = 10 #1.25
timestepper = d3.RK222
mesh = [16, 16]
dealias = 3/2
dtype = np.float64

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
V = basis.volume

# Fields
p = dist.Field(name='p', bases=basis)
T = dist.Field(name='T', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=s2_basis)
tau_T2 = dist.Field(name='tau_T2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)

# Substitutions
phi, theta, r = dist.local_grids(basis)
ez = dist.VectorField(coords, bases=basis)
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)
#f = d3.Grid(2*ez/Ekman)
f = 2*ez/Ekman
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r/Ro
lift_basis = basis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1) # First-order reduction

# Problem
problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - div(grad_T)/Prandtl + lift(tau_T2) = - (u@grad(T))")
problem.add_equation("dt(u) - div(grad_u) + grad(p)/Ekman - Rayleigh*rvec*T/Ekman + lift(tau_u2) = cross(u, curl(u) + f)")
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# for testing
solver.stop_iteration = 100
# Initial conditions
A = 0.1
x = 2*r-Ri-Ro
T['g'] = Ri*Ro/r - Ri + 210*A/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

out_cadence = 1e-2
# Analysis
snapshots = solver.evaluator.add_file_handler('slices', sim_dt=1e-2, max_writes=10)
snapshots.add_task(T(r=Ro), scales=dealias, name='T_r_outer')
snapshots.add_task(T(r=Ri), scales=dealias, name='T_r_inner')
snapshots.add_task(T(r=(Ri+Ro)/2), scales=dealias, name='T_r_mid')
snapshots.add_task(T(phi=0), scales=dealias, name='T_phi_start')
snapshots.add_task(T(phi=3*np.pi/2), scales=dealias, name='T_phi_end')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=out_cadence, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(T(r=(Ri+Ro)/2,theta=np.pi/2), name='T_profile')

traces = solver.evaluator.add_file_handler('traces', sim_dt=out_cadence, max_writes=100)
traces.add_task(0.5*d3.integ(u@u)/V, name='KE')

# CFL
max_timestep = 1e-4
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u), name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

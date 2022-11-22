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
MPrandtl = 5
Rayleigh = 100
Ri = 7/13
Ro = 20/13
Nr = 128
Lmax = 63
Ntheta = Lmax + 1
Nphi = 2*Ntheta
stop_sim_time = 15 #1.25
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
A = dist.VectorField(coords, name='A', bases=basis)
Phi = dist.Field(name='Phi', bases=basis)
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=s2_basis)
tau_T2 = dist.Field(name='tau_T2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)
tau_A1 = dist.VectorField(coords, name='tau_A1', bases=s2_basis)
tau_A2 = dist.VectorField(coords, name='tau_A2', bases=s2_basis)
tau_Phi = dist.Field(name='tau_Phi')

# Substitutions
phi, theta, r = dist.local_grids(basis)
ez = dist.VectorField(coords, bases=basis)
ez['g'][2] = np.cos(theta)
ez['g'][1] = -np.sin(theta)
f = d3.Grid(2*ez/Ekman)
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r/Ro
er = dist.VectorField(coords)
er['g'][2] = 1
B = d3.curl(A)
lift_basis = basis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1) # First-order reduction
grad_A = d3.grad(A) + rvec*lift(tau_A1) # First-order reduction
ell_func_o = lambda ell: ell+1
A_potential_bc_o = d3.radial(d3.grad(A)(r=Ro)) + d3.SphericalEllProduct(A, coords, ell_func_o)(r=Ro)/Ro
ell_func_i = lambda ell: -ell
A_potential_bc_i = d3.radial(d3.grad(A)(r=Ri)) + d3.SphericalEllProduct(A, coords, ell_func_i)(r=Ri)/Ri

# Problem
problem = d3.IVP([p, T, u, A, Phi, tau_p, tau_T1, tau_T2, tau_u1, tau_u2, tau_A1, tau_A2, tau_Phi], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - div(grad_T)/Prandtl + lift(tau_T2) = - (u@grad(T))")
problem.add_equation("dt(u) - div(grad_u) + grad(p)/Ekman - Rayleigh*rvec*T/Ekman + lift(tau_u2) = cross(u, curl(u) + f) + cross(B, lap(A))/MPrandtl/Ekman")
problem.add_equation("trace(grad_A) + tau_Phi = 0")
problem.add_equation("dt(A) - div(grad_A)/MPrandtl + grad(Phi) + lift(tau_A2) = cross(u, B)")
problem.add_equation("T(r=Ri) = 1")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("A_potential_bc_i = 0")
problem.add_equation("T(r=Ro) = 0")
problem.add_equation("u(r=Ro) = 0")
problem.add_equation("A_potential_bc_o = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("integ(Phi) = 0") # Scalar potential gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
amp = 0.1
x = 2*r-Ri-Ro
T['g'] = Ri*Ro/r - Ri + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

# We want to solve for an initial vector potential A
# with curl(A) = B0. We will do this as a BVP.
B0 = dist.VectorField(coords, bases=basis)
B0['g'][0] = 5*np.sin(np.pi*(r-Ri))*np.sin(2*theta)
B0['g'][1] = 5/8 * (9*r - 8*Ro - Ri**4/r**3)*np.sin(theta)
B0['g'][2] = 5/8 * (8*Ro - 6*r - 2*Ri**4/r**3)*np.cos(theta)
tau_Phi1 = dist.Field(bases=s2_basis)

mag_BVP = d3.LBVP([A, Phi, tau_A1, tau_Phi1, tau_Phi], namespace=locals())
mag_BVP.add_equation("curl(A) + grad(Phi) + lift(tau_A1) = B0")
mag_BVP.add_equation("div(A) + lift(tau_Phi1) + tau_Phi = 0")
mag_BVP.add_equation("angular(A_potential_bc_o) = 0", condition='ntheta!=0')
mag_BVP.add_equation("angular(A_potential_bc_i) = 0", condition='ntheta!=0')
mag_BVP.add_equation("radial(A_potential_bc_o) = 0", condition='ntheta==0')
mag_BVP.add_equation("radial(A_potential_bc_i) = 0", condition='ntheta==0')
mag_BVP.add_equation("integ(Phi) = 0")
solver_BVP = mag_BVP.build_solver()
solver_BVP.solve()

# Analysis
snapshots = solver.evaluator.add_file_handler('slices', sim_dt=1e-2, max_writes=10)
snapshots.add_task(T(r=Ro), scales=dealias, name='T_r_outer')
snapshots.add_task(T(r=Ri), scales=dealias, name='T_r_inner')
snapshots.add_task(T(r=(Ri+Ro)/2), scales=dealias, name='T_r_mid')
snapshots.add_task(T(phi=0), scales=dealias, name='T_phi_start')
snapshots.add_task(T(phi=3*np.pi/2), scales=dealias, name='T_phi_end')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=1e-3, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(B(r=(Ri+Ro)/2,theta=np.pi/2), name='B_profile')
profiles.add_task(T(r=(Ri+Ro)/2,theta=np.pi/2), name='T_profile')

traces = solver.evaluator.add_file_handler('traces', sim_dt=1e-3, max_writes=100)
traces.add_task(0.5*d3.integ(u@u)/V, name='KE')
traces.add_task(0.5*d3.integ(B@B)/V/Ekman/MPrandtl, name='ME')

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


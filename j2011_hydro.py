"""
Dedalus script for shell anelastic convection,
based on Jones et al 2011 convective benchmark case 0.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# parameters
beta = 0.35
Ekman = 1e-3
Prandtl = 1
Rayleigh = 351806
n = 2
Nrho = 5

dealias = 3/2
#Nr = 128
#Lmax = 255
Nr = 128
Lmax = 127
Ntheta = Lmax+1
Nphi = 2*Ntheta
timestepper = d3.RK222
stop_sim_time = 3
dtype = np.float64
mesh = [16,16] #[32, 32]

Ro = 1/(1-beta)
Ri = Ro - 1

zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
zeta_in  = (1 + beta - zeta_out) / beta
c0 = (2*zeta_out - beta - 1) / (1 - beta)
c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2
Di = c1*Prandtl/Rayleigh

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
V = basis.volume

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
S = dist.Field(name='S', bases=basis)
tau_p = dist.Field(name='tau_p')
tau_S1 = dist.Field(name='tau_T1', bases=s2_basis)
tau_S2 = dist.Field(name='tau_T2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)

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
f = d3.Grid(2*ez/Ekman)
omega = d3.curl(u)
g = dist.VectorField(coords, bases=basis.radial_basis)
g['g'][2] = 1/r**2
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r
zeta = dist.Field(bases=basis.radial_basis)
zeta['g'] = c0 + c1/r
rho0 = zeta**n
p0 = zeta**(n+1)
grad_log_rho0 = d3.grad(np.log(rho0))
grad_log_p0 = d3.grad(np.log(p0))
lift_basis = basis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
grad_S = d3.grad(S) + rvec*lift(tau_S1) # First-order reduction
#stress = rho0*(grad_u + d3.trans(grad_u))
I = dist.TensorField(coords, name='I', bases=basis.radial_basis)
for i in range(3):
    I['g'][i,i] = 1
stress = grad_u + d3.trans(grad_u) - 2/3*d3.trace(grad_u)*I
strain = 1/2*(d3.grad(u) + d3.trans(d3.grad(u)))
viscous_heating = 2*(d3.trace(strain @ strain) - 1/3*d3.div(u)**2)
zetag = d3.Grid(zeta)

# Problem
problem = d3.IVP([p, S, u, tau_p, tau_S1, tau_S2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + u@grad_log_rho0 + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - (div(stress) + stress@grad_log_rho0) - Rayleigh/Prandtl*S*g + lift(tau_u2) = cross(u, omega + f)")
problem.add_equation("dt(S) - (div(grad_S) + grad_S@grad_log_p0)/Prandtl + lift(tau_S2) = - (u@grad(S)) + Di/zetag*viscous_heating")
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
snapshots = solver.evaluator.add_file_handler('slices', sim_dt=1e-2, max_writes=10)
snapshots.add_task(S(r=Ro), scales=dealias, name='S_r_outer')
snapshots.add_task(S(r=Ri), scales=dealias, name='S_r_inner')
snapshots.add_task(S(r=(Ri+Ro)/2), scales=dealias, name='S_r_mid')
snapshots.add_task(S(phi=0), scales=dealias, name='S_phi_start')
snapshots.add_task(S(phi=3*np.pi/2), scales=dealias, name='S_phi_end')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=1e-3, max_writes=100)
profiles.add_task(u(r=(Ri+Ro)/2,theta=np.pi/2), name='u_profile')
profiles.add_task(S(r=(Ri+Ro)/2,theta=np.pi/2), name='S_profile')

traces = solver.evaluator.add_file_handler('traces', sim_dt=1e-3, max_writes=100)
traces.add_task(0.5*d3.integ(rho0*u@u), name='KE')
traces.add_task(d3.integ(rho0*d3.cross(rvec,u)@ex), name='Lx')
traces.add_task(d3.integ(rho0*d3.cross(rvec,u)@ey), name='Ly')
traces.add_task(d3.integ(rho0*d3.cross(rvec,u)@ez), name='Lz')
sphere_integ = lambda A: d3.Average(A, coords.S2coordsys)*4*np.pi
traces.add_task(-1/Prandtl*zeta_out**(n+1)*Ro**2*sphere_integ(d3.radial(d3.grad(S)(r=Ro))), name='Luminosity')

# CFL
max_timestep = 1e-4
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.35, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
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

"""
Dedalus script for testing ncc expansion sizes of angular-momentum conserving weight function for shells.

Usage:
    plot_L_cons_ncc.py [options]

Options:
    --Nr=<Nr>               Radial coeffs  [default: 128]
    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])

for system in ['matplotlib', 'h5py']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

from docopt import docopt
args = docopt(__doc__)

import numpy as np
import dedalus.public as de

Nr = int(args['--Nr'])
Ntheta = 4
Nphi = 2*Ntheta

dealias = 3/2
dtype = np.float64

Ri = r_inner = 7/13
Ro = r_outer = 20/13

# Bases
coords = de.SphericalCoordinates('phi', 'theta', 'r')
dist = de.Distributor(coords, dtype=dtype)
basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)

ncc_cutoff = float(args['--ncc_cutoff'])
b_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)

phi, theta, r = dist.local_grids(basis)

L_cons_ncc = dist.Field(bases=b_ncc, name='L_cons_ncc')
R_avg = (Ro+Ri)/2
L_cons_ncc['g'] = (r/R_avg)**3*np.sqrt((r/Ro-1)*(1-r/Ri))

logger.info("NCC expansions:")
for ncc in [L_cons_ncc]:
    logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.abs(L_cons_ncc['c'][0,0,:]))
ax.set_ylabel(r'$|L_c|$')
ax.set_yscale('log')
fig.savefig('L_cons_ncc_Nr{:d}.png'.format(Nr), dpi=300)

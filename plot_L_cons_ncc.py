"""
Dedalus script for testing ncc expansion sizes of angular-momentum conserving weight function for shells.

currently implemented benchmarks are:
    c2000       Christensen et al 2000, case 0, Boussinesq
    j2011       Jones et al 2011, hydro case, anelastic

Usage:
    plot_L_cons_ncc.py [options]

Options:
    --benchmark=<bench>  Benchmark to test [default: c2000]

    --Nr=<Nr>            Radial coeffs  [default: 128]
    --ncc_cutoff=<ncc>   Amplitude to truncate NCC terms [default: 1e-10]
"""
import numpy as np
import dedalus.public as de

dealias = 3/2
dtype = np.float64

def calculate_ncc(Nr, benchmark='c2000'):
    if benchmark == 'c2000':
        # Christensen et al. 2000, case 0 benchmark
        Ri = 7/13
        Ro = 20/13
    elif benchmark == 'j2011':
        # Jones et al. benchmark, hydro
        beta = 0.35
        Ro = 1/(1-beta)
        Ri = Ro - 1
    else:
        raise ValueError("benchmark={} is not either 'c2000' or 'j2011'".format(benchmark))

    Ntheta = 4
    Nphi = 2*Ntheta

    # Bases
    coords = de.SphericalCoordinates('phi', 'theta', 'r')
    dist = de.Distributor(coords, dtype=dtype)
    basis = de.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)

    phi, theta, r = dist.local_grids(basis)

    b_ncc = de.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
    #b_ncc = basis.radial_basis
    ρ = dist.Field(bases=b_ncc, name='ρ')

    if benchmark == 'c2000':
        ρ['g'] = 1

    elif benchmark == 'j2011':
        n = 2
        Nrho = 5

        zeta_out = (beta + 1) / ( beta*np.exp(Nrho/n) + 1 )
        zeta_in  = (1 + beta - zeta_out) / beta
        c0 = (2*zeta_out - beta - 1) / (1 - beta)
        c1 = (1 + beta)*(1 - zeta_out) / (1 - beta)**2

        zeta = dist.Field(bases=b_ncc, name='zeta')
        zeta['g'] = c0 + c1/r
        ρ.change_scales(dealias)
        ρ['g'] = (zeta**n).evaluate()['g']
        ρ.change_scales(1)

    L_cons_ncc = dist.Field(bases=b_ncc, name='L_cons_ncc')
    R_avg = (Ro+Ri)/2
    L_cons_ncc['g'] = ρ['g']*(r/R_avg)**3*np.sqrt((r/Ro-1)*(1-r/Ri))

    return L_cons_ncc

if __name__=='__main__':
    import logging
    logger = logging.getLogger(__name__.split('.')[-1])

    for system in ['matplotlib', 'h5py']:
        dlog = logging.getLogger(system)
        dlog.setLevel(logging.WARNING)

    from docopt import docopt
    args = docopt(__doc__)

    benchmark = args['--benchmark']

    Nr_min = 16
    Nr_max = int(args['--Nr'])

    def log_set(N_min, N_max):
        log2_N_min = int(np.log2(N_min))
        log2_N_max = int(np.log2(N_max))
        return np.logspace(log2_N_min, log2_N_max, base=2, num=(log2_N_max-log2_N_min+1), dtype=int)

    Nr_set = log_set(Nr_min, Nr_max)

    L_ncc_set = []
    for Nr in Nr_set:
        L_ncc_set.append(calculate_ncc(Nr, benchmark=benchmark))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ncc_cutoff = float(args['--ncc_cutoff'])

    n_layers = len(Nr_set)
    logger.info("NCC expansions:")
    i = 0
    for Nr, ncc in zip(Nr_set, L_ncc_set):
        logger.info("{}: {} (vs Nr={})".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape, Nr))
        ax.plot(np.abs(ncc['c'][0,0,:]), label='Nr={:d}'.format(Nr), alpha=0.75, zorder=n_layers-i)
        i+=1

    ax.set_ylabel(r'$|L_c|$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    fig.savefig('L_cons_ncc_{}_Nr{:d}-{:d}.png'.format(benchmark, Nr_min,Nr_max), dpi=300)

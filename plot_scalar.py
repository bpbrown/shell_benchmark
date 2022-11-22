"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

f = h5py.File(file, 'r')
data = {}
t = f['scales/sim_time'][:]
data_slice = (slice(None),0,0,0)
for key in f['tasks']:
    data[key] = f['tasks/'+key][data_slice]
f.close()

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

if 'ME' in data:
    energy_keys = ['KE', 'ME']
else:
    energy_keys = ['KE']


fig_E, ax_E = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    i_ten = int(0.9*data[key].shape[0])
    avg_E_f = np.mean(data[key][i_ten:])

    ax_E[0].plot(t, data[key], label=key)
    ax_E[1].plot(t, np.abs(data[key]-avg_E_f), label=key)

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
ax_E[1].set_xlabel('time')
ax_E[0].set_ylabel('energy density')
ax_E[1].set_ylabel(r'|E - E(t=t$_f$)|')
ax_E[1].legend(loc='lower left')
ax_E[1].set_yscale('log')
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/energies.png'.format(str(output_path)), dpi=300)

fig_tau, ax_tau = plt.subplots(nrows=2, sharex=True)
for i in range(2):
    p = ax_tau[i].plot(t, data['τ_u1'], label=r'$\tau_{u1,2}$')
    ax_tau[i].plot(t, data['τ_u2'], color=p[0].get_color(), linestyle='dashed')
    if 'τ_T1' in data:
        p = ax_tau[i].plot(t, data['τ_T1'], label=r'$\tau_{T1,2}$')
        ax_tau[i].plot(t, data['τ_T2'], color=p[0].get_color(), linestyle='dashed')
    elif 'τ_S1' in data:
        p = ax_tau[i].plot(t, data['τ_S1'], label=r'$\tau_{S1,2}$')
        ax_tau[i].plot(t, data['τ_S2'], color=p[0].get_color(), linestyle='dashed')
    ax_tau[i].plot(t, data['τ_p'], label=r'$\tau_{p}$')
    if 'τ_A1' in data:
        p = ax_tau[i].plot(t, data['τ_A1'], label=r'$\tau_{A1,2}$')
        ax_tau[i].plot(t, data['τ_A2'], color=p[0].get_color(), linestyle='dashed')
        #ax_tau[i].plot(t, data['τ_φ'], label=r'$\tau_{\phi}$')
for ax in ax_tau:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_ylabel(r'$<\tau>$')
    ax.legend()
ax_tau[1].set_xlabel('time')
ax_tau[1].set_yscale('log')
ylims = ax_tau[1].get_ylim()
ax_tau[1].set_ylim(max(1e-14, ylims[0]), ylims[1])
fig_tau.savefig('{:s}/tau_error.pdf'.format(str(output_path)))
fig_tau.savefig('{:s}/tau_error.png'.format(str(output_path)), dpi=300)

fig_L, ax_L = plt.subplots(nrows=2, sharex=True)
ax_L[0].plot(t, data['Lx'], label='Lx')
ax_L[0].plot(t, data['Ly'], label='Ly')
ax_L[0].plot(t, data['Lz'], label='Lz')
ax_L[1].plot(t, np.abs(data['Lx']), label='Lx')
ax_L[1].plot(t, np.abs(data['Ly']), label='Ly')
ax_L[1].plot(t, np.abs(data['Lz']), label='Lz')

for ax in ax_L:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_ylabel('Angular Momentum')
    ax.legend(loc='lower left')
ax_L[1].set_xlabel('time')
ax_L[1].set_yscale('log')
fig_L.savefig('{:s}/angular_momentum.pdf'.format(str(output_path)))
fig_L.savefig('{:s}/angular_momentum.png'.format(str(output_path)), dpi=300)

fig_f, ax_f = plt.subplots(nrows=2)
for ax in ax_f:
    ax.plot(t, data['Re'], label='Re')
    ax_r = ax.twinx()
    ax_r.plot(t, data['Ro'], label='Ro', color='tab:orange')
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('fluid parameters')
    ax.legend(loc='lower left')

ax_f[1].set_yscale('log')
ax_r.set_yscale('log') # relies on it being the last instance; poor practice
fig_f.savefig('{:s}/Re_and_Ro.pdf'.format(str(output_path)))
fig_f.savefig('{:s}/Re_and_Ro.png'.format(str(output_path)), dpi=300)

benchmark_set = ['KE', 'Ro', 'Re', 'τ_u1', 'τ_u2', 'τ_p']
if 'ME' in data:
    benchmark_set = [benchmark_set[0], 'ME'] + benchmark_set[1:]
    benchmark_set = benchmark_set[0:-1] + ['τ_A1', 'τ_A2'] + [benchmark_set[-1]]
if 'τ_T1' in data:
    benchmark_set = benchmark_set[0:-1] + ['τ_T1', 'τ_T2'] + [benchmark_set[-1]]
elif 'τ_S1' in data:
    benchmark_set = benchmark_set[0:-1] + ['τ_S1', 'τ_S2'] + [benchmark_set[-1]]

i_ten = int(0.9*data[benchmark_set[0]].shape[0])
print("benchmark values")
for benchmark in benchmark_set:
    print("{:s} = {:14.12g} +- {:4.2g} (averaged from {:g}-{:g})".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:]), t[i_ten], t[-1]))

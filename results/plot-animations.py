import numpy as np
import matplotlib.pyplot as plt
import pickle
from folsom import Folsom
import pandas as pd
import seaborn as sns

snapshots = pickle.load(
    open('results/hist-opt/snapshots-opt-depth-5-seed-20.pkl', 'rb'))
# P = snapshots['best_P'][-1]
save_path = 'figs/anim/img'


def init_plotting(w, h):
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (w, h)
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


def colortsplot(df, filename, dpi):

    init_plotting(8.5, 3)

    colors = {'Release_Demand': 'cornsilk',
              'Hedge_90': 'indianred',
              'Hedge_80': 'indianred',
              'Hedge_70': 'indianred',
              'Hedge_60': 'indianred',
              'Hedge_50': 'indianred',
              'Flood_Control': 'lightsteelblue'}

    df.storage.plot(color='0.6', linewidth=2)
    df.Ss.plot(color='k', linewidth=2, zorder=10)

    for pol in set(df.policy):
        first = df.index[(df.policy == pol) & (df.policy.shift(1) != pol)]
        last = df.index[(df.policy == pol) & (df.policy.shift(-1) != pol)]

        for f, l in zip(first, last):
            plt.axvspan(f, l + pd.Timedelta('1 day'),
                        facecolor=colors[pol], edgecolor='none', alpha=0.4)

    plt.legend(['Observed', 'Policy Tree'], loc=3, ncol=2)
    plt.ylim([0, 1000])
    plt.ylabel('Storage (TAF)')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


max_nfe = 15000

for i, P in enumerate(snapshots['best_P']):

    nfe = snapshots['nfe'][i]
    print str(P)
    print nfe

    if nfe > max_nfe:
        break

    # first, the tree
    nfestring = 'nfe-' + '%06d' % nfe + '.png'
    P.graphviz_export(save_path + '/tree-' + nfestring, dpi=150)

    # then, the time series
    model = Folsom('folsom/data/folsom-daily.csv',
                   sd='1995-10-01', ed='2015-09-30', use_tocs=False)
    df = model.f(P, mode='simulation')
    colortsplot(df, filename=save_path + '/folsom-' + nfestring, dpi=150)

    # last, the objective function
    init_plotting(5, 5)
    plt.plot(snapshots['nfe'][:i + 1], snapshots['best_f']
             [:i + 1], linewidth=2, color='steelblue')
    plt.xlim([0, max_nfe])
    plt.ylim([0, np.max(snapshots['best_f'])])
    plt.ylabel('J (TAF/d)$^2$')
    plt.xlabel('NFE')
    plt.tight_layout()
    plt.savefig(save_path + '/convergence-' + nfestring, dpi=150)
    plt.close()

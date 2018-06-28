import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns


def water_day(d):
    return d - 274 if d >= 274 else d + 91


def cfs_to_taf(Q):
    return Q * 2.29568411 * 10**-5 * 86400 / 1000


def taf_to_cfs(Q):
    return Q * 1000 / 86400 * 43560


def init_plotting():
    sns.set_style('whitegrid')

    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


def plot_results(df, filename=None, dpi=300):

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    df.storage.plot(ax=ax0, color='k')
    df.Ss.plot(ax=ax0, color='indianred')
    ax0.set_ylabel('')
    ax0.xaxis.grid(False)
    ax0.set_title('Storage, TAF', family='OfficinaSanITCMedium', loc='left')
    ax0.legend(['Observed', 'Simulated'], loc=3, ncol=3)
    ax0.set_ylim([0, 1000])

    ax1 = plt.subplot(gs[1])
    ax1.scatter(df.storage.values, df.Ss.values, s=3,
                c='steelblue', edgecolor='none', alpha=0.7)
    ax1.set_xlim([0, 1000])
    ax1.set_ylim([0, 1000])
    ax1.set_ylabel('Simulated (TAF)')
    ax1.set_xlabel('Observed (TAF)')
    ax1.plot([0, 1000], [0, 1000], color='k')
    ax1.annotate('$R^2 = %0.2f$' % np.corrcoef(
        df.Ss.values, df.storage.values)[0, 1], xy=(600, 100), color='0.3')

    ax2 = plt.subplot(gs[2])
    np.log10(taf_to_cfs(df.outflow)).plot(color='k', ax=ax2)
    np.log10(taf_to_cfs(df.Rs)).plot(color='indianred', ax=ax2)
    # taf_to_cfs(df.inflow).plot(color='blue')
    # df.demand.plot(color='green', linewidth=2)
    ax2.set_ylabel('')
    ax2.xaxis.grid(False)
    ax2.set_title('Release, log$_{10}$ cfs',
                  family='OfficinaSanITCMedium', loc='left')
    ax2.legend(['Observed', 'Simulated'], loc=3, ncol=3)
    ax2.set_ylim([1.5, 5.5])

    ax3 = plt.subplot(gs[3])
    r2 = np.corrcoef(df.Rs.values, df.outflow.values)[0, 1]
    ax3.scatter(np.log10(taf_to_cfs(df.outflow)), np.log10(
        taf_to_cfs(df.Rs)), s=3, c='steelblue', edgecolor='none', alpha=0.7)
    ax3.set_xlim([2.5, 5.5])
    ax3.set_ylim([2.5, 5.5])
    ax3.plot([1.0, 5.5], [1.0, 5.5], color='k')
    ax3.set_ylabel('Simulated (log cfs)')
    ax3.set_xlabel('Observed (log cfs)')
    ax3.annotate('$R^2 = %0.2f$' % r2, xy=(4.25, 2.75), color='0.3')

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=dpi)
        plt.close()


init_plotting()

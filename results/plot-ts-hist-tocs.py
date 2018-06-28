import numpy as np
import matplotlib.pyplot as plt
import pickle
from ptreeopt.opt import *
from folsom import Folsom
import pandas as pd
import seaborn as sns


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



# snapshots = pickle.load(open('results/hist-fit-fp/snapshots-fit-hist.pkl', 'rb'))
snapshots = pickle.load(
    open('results/hist-tocs/snapshots-tocs-depth-4-seed-23.pkl', 'rb'))
# snapshots = pickle.load(open('snapshots-hist-tocs-test.pkl', 'rb'))

# ccs = 'bcc-csm1-1_rcp26_r1i1p1'
# snapshots = pickle.load(open('results/cc-opt/snapshots-cc-' + ccs + '.pkl', 'rb'))
# snapshots = pickle.load(open('results/snapshots-opt-hist.pkl', 'rb'))
# snapshots = pickle.load(open('results/hist-opt/snapshots-depth-4-seed-8.pkl', 'rb'))

model = Folsom('folsom/data/folsom-daily-w2016.csv',
               sd='1995-10-01', ed='2016-09-30', use_tocs=True)
# model = Folsom('data/folsom-cc-inflows.csv', sd='1999-10-01', ed='2099-09-30',
# scenario = ccs, cc = True)

P = snapshots['best_P'][-1]
print(snapshots['best_f'][-1])
P.graphviz_export('hist-tocs-depth-4-seed-23.svg')

print P
df = model.f(P, mode='simulation')
df.policy.ix[0] = df.policy.ix[1]

init_plotting(10, 3)

# colors = ['cornsilk', 'lightsteelblue', 'indianred']
df.storage.plot(color='0.8', linewidth=3)
df.Ss.plot(color='steelblue', linewidth=1, zorder=10)

# print np.corrcoef(df.storage.values, df.Ss.values)
# df['shortage'] = df.demand - df.outflow
# df.shortage[df.shortage < 0] = 0
# print (df.shortage**2).mean()

# include the SDP/DDP results
# careful, matteo's results don't have leap years. and don't use the last value
dp_t = df.index[~((df.index.month == 2) & (df.index.day == 29))]
ddp = pd.read_csv('results/ddp-sdp/resultDDP.txt',
                  delim_whitespace=True, usecols=[0], header=None)
df['ddp'] = pd.Series(ddp[0].values, index=dp_t)
sdp = pd.read_csv('results/ddp-sdp/resultSDP.txt',
                  delim_whitespace=True, usecols=[0], header=None)
df['sdp'] = pd.Series(sdp[0].values, index=dp_t)
df.ddp.plot(color='r', linewidth=1)
df.sdp.plot(color='0.3', linewidth=1)

# for pol, c in zip(set(df.policy), colors):
#   first = df.index[(df.policy == pol) & (df.policy.shift(1) != pol)]
#   last = df.index[(df.policy == pol) & (df.policy.shift(-1) != pol)]

#   for f,l in zip(first,last):
#     plt.axvspan(f,l+pd.Timedelta('1 day'), facecolor=c, edgecolor='none', alpha=0.6)


plt.title('(c) Storage Trajectory, TAF',
          family='OfficinaSanITCMedium', loc='left')
plt.legend(['Observed', 'Tree', 'DDP', 'SDP'], loc=3,
           ncol=4, bbox_to_anchor=(0.35, 0.97))
plt.ylim([0, 1000])


# plt.scatter(df.outflow.values, df.outflow.shift(-1).values)
# plt.scatter(df.Rs.values, df.Rs.shift(-1).values, color='r')
# plt.show()
plt.savefig('hist-tocs-ts.svg')

# plt.savefig('figs/best-historical-ts-policy.svg')

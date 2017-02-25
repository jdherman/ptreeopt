import numpy as np
import matplotlib.pyplot as plt
import pickle
from ptreeopt import PTreeOpt
from folsom import Folsom
import pandas as pd
import seaborn as sns

def init_plotting(w,h):
  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (w,h)
  plt.rcParams['font.size'] = 13
  plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']



names = ['access1-0_rcp45_r1i1p1',
         'canesm2_rcp85_r1i1p1',
         'giss-e2-h-cc_rcp45_r1i1p1',
         'gfdl-esm2g_rcp45_r1i1p1',
         'miroc-esm_rcp85_r1i1p1']


# snapshots = pickle.load(open('results/hist-fit-fp/snapshots-fit-hist.pkl', 'rb'))
# snapshots = pickle.load(open('results/hist-tocs/snapshots-tocs-depth-3-seed-0.pkl', 'rb'))
snapshots = pickle.load(open('results/cc-full/snapshots-cc-full-4.pkl', 'rb'))

# ccs = 'bcc-csm1-1_rcp26_r1i1p1'
# snapshots = pickle.load(open('results/cc-opt/snapshots-cc-' + ccs + '.pkl', 'rb'))
# snapshots = pickle.load(open('results/snapshots-opt-hist.pkl', 'rb'))
# snapshots = pickle.load(open('results/hist-opt/snapshots-depth-4-seed-8.pkl', 'rb'))

# model = Folsom('folsom-daily.csv', sd='1995-10-01', ed='2015-09-30', use_tocs = False)
model = Folsom('folsom/data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
               cc = True, use_tocs=False)

P = snapshots['best_P'][-1]
print P

model.set_scenario(names[4])
df = model.f(P, mode='simulation')
df.policy.ix[0] = df.policy.ix[1]


init_plotting(8.5,3)

colors = {'Release_Demand': 'cornsilk', 
          'Hedge_90': 'indianred', 
          'Hedge_80': 'indianred', 
          'Hedge_70': 'indianred', 
          'Hedge_60': 'indianred', 
          'Hedge_50': 'indianred', 
          'Flood_Control': 'lightsteelblue'}


# df.storage.plot(color='0.6', linewidth=2)
df.Ss.plot(color='k', linewidth=2, zorder=10)

# # print np.corrcoef(df.storage.values, df.Ss.values)

print set(df.policy)

for pol in set(df.policy):
  first = df.index[(df.policy == pol) & (df.policy.shift(1) != pol)]
  last = df.index[(df.policy == pol) & (df.policy.shift(-1) != pol)]

  for f,l in zip(first,last):
    plt.axvspan(f,l+pd.Timedelta('1 day'), facecolor=colors[pol], edgecolor='none', alpha=0.4)


# plt.title('Folsom Reservoir Storage, TAF', family='OfficinaSanITCMedium', loc='left')
# plt.legend(['Observed (J = 0.29)', 'Tree (J = 0.03)'], loc=3, ncol=4)
plt.ylim([0,1000])

# # plt.scatter(df.outflow.values, df.outflow.shift(-1).values)
# # plt.scatter(df.Rs.values, df.Rs.shift(-1).values, color='r')
plt.tight_layout()
plt.show()
# plt.savefig('hist-opt-ts-flood.svg')




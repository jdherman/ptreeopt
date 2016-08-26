import numpy as np
import matplotlib.pyplot as plt
import pickle
from opt import *
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


snapshots = pickle.load(open('results/hist-fit-fp/snapshots-fit-historical.pkl', 'rb'))
model = Folsom('folsom-daily.csv', sd='1995-10-01', ed='2015-09-30', fit_historical = False)

P = snapshots['best_P'][-1]
print P
df = model.f(P, mode='simulation')
df.policy.ix[0] = df.policy.ix[1]

init_plotting(10,3)

colors = ['cornsilk', 'lightsteelblue', 'indianred']
df.storage.plot(color='0.6', linewidth=2)
df.Ss.plot(color='k', linewidth=2)

for pol, c in zip(set(df.policy), colors):
  first = df.index[(df.policy == pol) & (df.policy.shift(1) != pol)]
  last = df.index[(df.policy == pol) & (df.policy.shift(-1) != pol)]

  for f,l in zip(first,last):
    plt.axvspan(f,l+pd.Timedelta('1 day'), facecolor=c, edgecolor='none', alpha=0.6)


plt.title('Folsom Reservoir Storage, TAF', family='OfficinaSanITCMedium', loc='left')
plt.legend(['Observed', 'Simulated'], loc=3, ncol=3)
plt.ylim([0,1000])
# plt.show()

plt.savefig('figs/best-historical-ts-policy.svg')



import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from folsom import Folsom


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


init_plotting(7,6)

##############################

# d = []

scenarios = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0).columns
annQs = pd.read_csv('data/folsom-cc-annQ-MA30.csv', index_col=0, parse_dates=True)

# snapshots = pickle.load(open('results/hist-tocs/snapshots-tocs-depth-3-seed-0.pkl', 'rb'))
# # snapshots = pickle.load(open('results/hist-opt/snapshots-opt-depth-5-seed-20.pkl', 'rb'))
# histP = snapshots['best_P'][-1]
# print scenarios
# # first get cc-opt results
# # then get the re-evaluated historical policy on the CC scenarios

# for s in scenarios:
#   if 'rcp26' not in s:
#     print s
#     df = pickle.load(open('results/cc-opt/snapshots-cc-' + s + '.pkl', 'rb'))

#     f = df['best_f'][-1]

#     model = Folsom('data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
#                 scenario = s, cc = True, use_tocs=True)

#     Jv = model.f(histP)

#     d.append({'scenario': s, 'type': 'Adapted', 'J': f})
#     d.append({'scenario': s, 'type': 'Baseline', 'J': Jv})
#     print pd.DataFrame(d)

# print d
# pickle.dump(d, open('cc-hist-val.pkl', 'w'))

d = pickle.load(open('cc-hist-val.pkl', 'rb'))
df = pd.DataFrame(d)
df.J[df.J > 10] /= 1000
annQs = annQs[df.scenario.unique()]['2099'].max(axis=0).sort_values()
sns.stripplot(data=df, x='J', y='scenario', hue='type', 
              order=annQs.index.values, split=False, edgecolor='none',
              palette=['r', 'k'])
plt.xscale('log')

for i,s in enumerate(annQs.index.values):
  points = df[df.scenario == s].J.values
  plt.plot(points, [i,i], color='0.75', linewidth=0.5)

plt.xlim([0.01,10000])
plt.ylim([-1,76])
plt.gca().set_yticklabels([])
plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(False)
sns.despine(left=False, bottom=False)
plt.axvline(10.0, color='k', linestyle='--', linewidth=1)

plt.tight_layout()
# plt.show()
plt.savefig('stripplot.svg')

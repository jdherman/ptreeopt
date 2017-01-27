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
  plt.rcParams['font.size'] = 8
  plt.rcParams['font.family'] = 'Source Sans Pro'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


init_plotting(10,10)

##############################

# d = []

# scenarios = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0).columns
annQs = pd.read_csv('folsom/data/folsom-cc-annQ-MA30.csv', index_col=0, parse_dates=True)
annQs['hist'] = 2700.0
annQs = annQs[annQs.columns.drop(annQs.filter(like='rcp26').columns)]
annQs = annQs['2099'].max(axis=0).sort_values()

# snapshots = pickle.load(open('results/hist-tocs/snapshots-tocs-depth-3-seed-0.pkl', 'rb'))
# # snapshots = pickle.load(open('results/hist-opt/snapshots-opt-depth-5-seed-20.pkl', 'rb'))
# histP = snapshots['best_P'][-1]
# # first get cc-opt results
# # then get the re-evaluated historical policy on the CC scenarios

# for s1 in scenarios:
#   print s1
#   if 'rcp26' not in s1:
#     df = pickle.load(open('results/cc-opt/snapshots-cc-' + s1 + '.pkl', 'rb'))

#     # The 1.0 value for this scenario
#     f = df['best_f'][-1]
#     d.append({'s_opt': s1, 's_eval': s1, 'Jratio': 1.0, 'J': f})

#     # run the historical policy in this scenario
#     model = Folsom('data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
#                 scenario = s1, cc = True, use_tocs=True)
#     Jv = model.f(histP)
#     d.append({'s_opt': 'hist', 's_eval': s1, 'Jratio': Jv/f, 'J': Jv})

#     # then run all the other CC policies on this scenario
#     for s2 in scenarios:
#       if ('rcp26' not in s2) and (s2 != s1):

#         snapshots = pickle.load(open('results/cc-opt/snapshots-cc-' + s2 + '.pkl', 'rb'))
#         ccP = snapshots['best_P'][-1]

#         Jv = model.f(ccP)
#         d.append({'s_opt': s2, 's_eval': s1, 'Jratio': Jv/f, 'J': Jv})

# pickle.dump(d, open('cc-val-all.pkl', 'w'))

d = pickle.load(open('results/validations/cc-val-all.pkl', 'rb'))
# print pd.DataFrame(d)
# d.append({'s_opt': 'hist', 's_eval': 'hist', 'Jratio': 1.0})
df = pd.DataFrame(d).pivot('s_opt', 's_eval', 'J')
cols = annQs.index.tolist()
df = df.reindex(cols)
cols.remove('hist')
df = df[cols[:]]
# print (df.values < 20) mask=(df.values>20)
ax = sns.heatmap(df, vmax=10, vmin=1, square=True)

yt = plt.gca().get_yticklabels()
plt.gca().set_yticklabels([thing.get_text().split('_r1i1p1')[0] for thing in yt])
plt.yticks(rotation=0)

xt = plt.gca().get_xticklabels()
plt.gca().set_xticklabels([thing.get_text().split('_r1i1p1')[0] for thing in xt])
plt.xticks(rotation=90)

plt.ylabel('Optimized in Scenario ...')
plt.xlabel('Evaluated in Scenario ...')
plt.tight_layout()
# plt.show()
plt.savefig('heatmap-validation.svg')

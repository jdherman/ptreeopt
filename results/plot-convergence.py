import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from matplotlib.ticker import ScalarFormatter
# http://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting

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


init_plotting(7,4)

path = 'hist-opt'
DDP = 0.06 # placeholder value for now
HIST = 0.2869 / DDP # (TAF/d)^2, normalized by ddp value
SDP = 0.22 / DDP
# Values from Matteo on 8/30/16 email


##############################
plt.subplot(1,2,1)

depth = 4
seeds = 50

for s in range(seeds):
  data = pickle.load(open(path + '/snapshots-depth-' + str(depth) + '-seed-' + str(s) + '.pkl', 'rb'))
  print s
  nfe = data['nfe']
  best_f = np.array(data['best_f'])
  print best_f[-1]
  print data['best_P'][-1] # to see tree logic
  plt.loglog(nfe, best_f / DDP, linewidth=0.5, color='steelblue')

plt.xlabel('NFE')
plt.ylabel('J / J$^*$')
plt.xlim([50,20000])
plt.title('(a) Convergence \n Max Depth = ' + str(depth), y=0.8, x=0.7, family='OfficinaSanITCMedium')

# annotations and lines
plt.axhline(HIST, linewidth=2, color='k')
plt.annotate('Observed', (3500, HIST+0.75), family='OfficinaSanITCBooIta')
plt.axhline(SDP, linewidth=2, color='k')
plt.annotate('SDP', (60, SDP-1.5), family='OfficinaSanITCBooIta')
plt.axhline(1.0, linewidth=3, color='r')
plt.annotate('DDP = 1.0', (60, 1.07), color='r', family='OfficinaSanITCBooIta')

for a in [plt.gca().xaxis, plt.gca().yaxis]:
  a.set_major_formatter(ScalarFormatter())


##############################
##############################

plt.subplot(1,2,2)
data_to_plot = []
# Jv = np.loadtxt('historical-validation/results.csv', delimiter=',')
d = []

for depth in range(1,6):
  Jend = np.zeros(50)

  for s in range(seeds):
    data = pickle.load(open(path + '/snapshots-depth-' + str(depth) + '-seed-' + str(s) + '.pkl', 'rb'))
    Jend = data['best_f'][-1] / DDP
    d.append({'depth': depth, 'seed': s, 'type': 'training', 'J': Jend})

    # the validation is not good right now. training overfits to floods.
    # d.append({'depth': depth, 'seed': s, 'type': 'validation', 'J': Jv[s,depth]})

  # data_to_plot.append(Jend)


df = pd.DataFrame(d)
flierprops = dict(marker='o', markersize=5)

# also want to show validation here as a grouped plot
sns.boxplot(data=df, x='depth', y='J', width=0.5, 
            saturation=0.5, color='steelblue') #hue='type' for validation
plt.gca().set_xticklabels(range(1,9))
plt.xlabel('Max Tree Depth')
plt.ylabel('J / J$^*$ after 20000 NFE')
plt.yscale('log')

# annotations and lines
plt.axhline(HIST, linewidth=2, color='k')
plt.annotate('Observed', (0, HIST+0.75), family='OfficinaSanITCBooIta')
plt.axhline(SDP, linewidth=2, color='k')
plt.annotate('SDP', (0, SDP-1), family='OfficinaSanITCBooIta')
plt.axhline(1.0, linewidth=3, color='r')
plt.annotate('DDP = 1.0', (-0.4, 0.8), color='r', family='OfficinaSanITCBooIta')

plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.tight_layout()
plt.show()
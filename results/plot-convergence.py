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


init_plotting(8,4)

path = 'results/hist-tocs'
# CHECK THESE
# DDP = 0.06
# HIST = 0.2869 # (TAF/d)^2
# SDP = 0.22 
# Values from Matteo on 8/30/16 email


# NEW VALUES FEB 2017
DDP = 0.1270
SDP = 0.3087
HIST = 0.3442

##############################
plt.subplot(1,2,1)

depth = 4
seeds = 50

for s in range(seeds):
  data = pickle.load(open(path + '/snapshots-tocs-depth-' + str(depth) + '-seed-' + str(s) + '.pkl', 'rb'))
  print s
  nfe = data['nfe']
  best_f = np.array(data['best_f'])
  print best_f[-1]
  print data['best_P'][-1] # to see tree logic
  plt.semilogx(nfe, best_f, linewidth=0.5, color='steelblue')

plt.xlabel('NFE')
plt.ylabel('J (TAF/d)$^2$')
plt.xlim([50,25000])
# plt.ylim([0.0,0.30])
plt.title('(a) Convergence, depth = ' + str(depth), loc='left', family='OfficinaSanITCMedium')

# annotations and lines
plt.axhline(HIST, linewidth=2, color='k')
plt.annotate('Observed = %0.2f' % HIST, (1002, HIST-0.02), family='OfficinaSanITCBooIta')
plt.axhline(SDP, linewidth=2, color='k')
plt.annotate('SDP = %0.2f' % SDP, (1002, SDP-0.02), family='OfficinaSanITCBooIta')
plt.axhline(DDP, linewidth=3, color='r')
plt.annotate('DDP = %0.2f' % DDP, (1002, DDP-0.02), color='r', family='OfficinaSanITCBooIta')

# for a in [plt.gca().xaxis, plt.gca().yaxis]:
#   a.set_major_formatter(ScalarFormatter())


##############################
##############################

plt.subplot(1,2,2)
data_to_plot = []
# Jv = np.loadtxt('historical-validation/results.csv', delimiter=',')
d = []

for depth in range(1,9):
  Jend = np.zeros(50)

  for s in range(seeds):
    data = pickle.load(open(path + '/snapshots-tocs-depth-' + str(depth) + '-seed-' + str(s) + '.pkl', 'rb'))
    Jend = data['best_f'][-1] 
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
plt.ylabel('J$_{end}$ (TAF/d)$^2$')
# plt.ylim([0.0,0.30])
# plt.yscale('log')
plt.title('(b) Reliability', loc='left', family='OfficinaSanITCMedium')

plt.ylim([0.10, 0.45])

# annotations and lines
plt.axhline(HIST, linewidth=2, color='k')
plt.annotate('Observed = %0.2f' % HIST, (3, HIST-0.02), family='OfficinaSanITCBooIta')
plt.axhline(SDP, linewidth=2, color='k')
plt.annotate('SDP = %0.2f' % SDP, (3, SDP-0.02), family='OfficinaSanITCBooIta')
plt.axhline(DDP, linewidth=3, color='r')
plt.annotate('DDP = %0.2f' % DDP, (3, DDP-0.02), color='r', family='OfficinaSanITCBooIta')

plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.tight_layout()
# plt.show()
plt.savefig('algo-results.svg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
from opt import *
import folsom
import pandas as pd

snapshots = pickle.load(open('snapshots-historical-13.pkl', 'rb'))

for i,P in enumerate(snapshots['best_P']):
  print str(P)
  nfestring = 'nfe-' + '%06d' % snapshots['nfe'][i] + '.png'
  P.graphviz_export('figs/anim/tree-' + nfestring, dpi=150)
  df = folsom.f(P, mode='simulation')
  folsom.plot_results(df, filename='figs/anim/folsom-' + nfestring, dpi=150)

  plt.rcParams['figure.figsize'] = (5, 5)
  plt.plot(snapshots['nfe'][:i+1], snapshots['best_f'][:i+1], linewidth=2, color='steelblue')
  plt.xlim([0,np.max(snapshots['nfe'])])
  plt.ylim([np.min(snapshots['best_f']), np.max(snapshots['best_f'])])
  plt.ylabel('RMSE')
  plt.xlabel('NFE')
  plt.savefig('figs/anim/convergence-' + nfestring, dpi=150)

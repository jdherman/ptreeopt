import numpy as np
import matplotlib.pyplot as plt
import pickle
from tree import *
# from opt import *
# import folsom
import pandas as pd


# Historical best fit

# snapshots = pickle.load(open('snapshots-historical-13.pkl', 'rb'))
# snapshots = pickle.load(open('hist-fit-fp/snapshots-fit-hist.pkl', 'rb'))
# P = snapshots['best_P'][-1]
# P.graphviz_export('../figs/fit-4-hist-results/hist-fit.svg')

# snapshots = pickle.load(open('hist-opt/snapshots-depth-4-seed-8.pkl', 'rb'))
# P = snapshots['best_P'][-1]
# P.graphviz_export('../figs/fig-4-hist-results/hist-opt.svg')


# CC selected trees
# want to highlight just a few
# for the scatter plot example
# names = ['access1-0_rcp45_r1i1p1',
#          'canesm2_rcp85_r1i1p1',
#          'giss-e2-h-cc_rcp45_r1i1p1',
#          'gfdl-esm2g_rcp45_r1i1p1',
#          'miroc-esm_rcp85_r1i1p1']
# for s in names:
#   data = pickle.load(open('cc-opt/snapshots-cc-' + s + '.pkl', 'rb'))

#   nfe = data['nfe']
#   best_f = np.array(data['best_f'])
#   P = data['best_P'][-1]
#   print '%s, %f, %s' % (s, best_f[-1], P) # to see tree logic
#   P.graphviz_export('../figs/fig-6-cc-results/tree-' + s + '.svg')


# cc-full tree (robust policy)
snapshots = pickle.load(open('cc-full/snapshots-cc-full-4.pkl', 'rb'))
P = snapshots['best_P'][-1]
print str(P)
P.graphviz_export('../figs/fig-7-cc-full-ts/cc-full-tree.svg')


# print P
# df = folsom.f(P, mode='simulation')
# df.policy.ix[0] = df.policy.ix[1]

# to show historical fit
# plt.figure(figsize=(10, 3))
# folsom.plot_results(df)

# def pos_sq(x):
#   return 0 if x < 0 else x**2

# cost = (df.demand - df.Rs).apply(pos_sq)
# print cost.mean()
# plt.ylabel('Daily Squared Deficit (TAF/d)^2')
# cost.plot()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pickle
from opt import *
import folsom
import pandas as pd

# snapshots = pickle.load(open('snapshots-historical-13.pkl', 'rb'))
snapshots = pickle.load(open('snapshots-test.pkl', 'rb'))

P = snapshots['best_P'][-1]
print P
df = folsom.f(P, mode='simulation')
df.policy.ix[0] = df.policy.ix[1]

# to show historical fit
# plt.figure(figsize=(10, 3)) 
# folsom.plot_results(df)

def pos_sq(x):
  return 0 if x < 0 else x**2

cost = (df.demand - df.Rs).apply(pos_sq)
print cost.mean()
plt.ylabel('Daily Squared Deficit (TAF/d)^2')
cost.plot()
plt.show()

# P.graphviz_export('figs/best.svg')
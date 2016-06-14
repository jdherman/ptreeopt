import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from opt import *

# simplified reservoir simulation

df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1955-10-01':'2015-09-30']

def f(P, plot=False):

  # Set some parameters
  K = 975 # capacity, TAF

  # data setup
  Q = df.inflow.values
  true_release = df.outflow.values
  doy = df.index.dayofyear
  T = len(Q)

  S = np.zeros(T)
  R = np.zeros(T)
  D = np.zeros(T)
  # deficit = 0
  ssq = 0

  S[0] = df.storage.values[0]

  for t in range(1,T):

    # demand
    if 120 < doy[t] < 270:
      D[t] = 7 # TAF/day
    else:
      D[t] = 3

    # release and mass balance
    W = S[t-1] + Q[t]
    R[t] = min(P.evaluate([W, doy[t]]), W)
    spill = max(W - R[t] - K, 0)
    S[t] = W - R[t] - spill

    # deficit += max(D[t] - R[t], 0)**2
    ssq += (R[t] + spill - true_release[t])**2

  if plot:
    plt.subplot(2,1,1)
    plt.plot(S, color='blue', linewidth=2)
    plt.plot(df.storage.values, color='cyan', linewidth=2)
    plt.subplot(2,1,2)
    plt.plot(R, color='red', linewidth=2)
    plt.plot(true_release, color='pink', linewidth=2)
    plt.plot(D, color='green', linewidth=2)
    plt.show()

  return ssq #deficit # minimize. If fitting historical.


# 2 features: water available, and month
feature_bounds = [[0,1000], [1,365]]
names = ['Storage', 'DOY']
action_bounds = [0, 228] # max of 115000 cfs

algorithm = PTreeOpt(f, 2000, feature_bounds, action_bounds, feature_names=names, 
                      population_size=100, log_frequency=100)

algorithm.initialize()
# print algorithm.objectives

algorithm.run()
print algorithm.best_P
print algorithm.best_f
# print [i for i in algorithm.population[0].L]
# f(algorithm.population[0], plot=True)
f(algorithm.best_P, plot=True)

algorithm.best_P.graphviz_export('graphviz/bestPfol')



# Need to add:


# feature variable names
# action variable name
# to make these work, they must be properties of the nodes

# discrete vs. continuous actions (it matters) -- not right now.
# operators (!!) mutation first

# to validate a tree ... this will be important
# lower nodes/logic (should not) contradict parent logic
# or else some branches will always evaluate false (this might be ok)





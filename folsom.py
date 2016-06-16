import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from opt import *

# simplified reservoir simulation

df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1980-10-01':'2015-09-30']

def water_day(d):
  return d - 274 if d >= 274 else d + 91

def f(P, mode='optimization'):

  # Set some parameters
  K = 975 # capacity, TAF

  # data setup
  Q = df.inflow.values
  true_release = df.outflow.values
  dowy = np.array([water_day(d) for d in df.index.dayofyear])
  T = len(Q)

  S,R,D,spill = [np.zeros(T) for _ in range(4)]

  cost = 0
  ssq = 0

  S[0] = df.storage.values[0]

  for t in range(1,T):

    # demand: hypothetical model
    # minimum of 3 TAF/day around January 9th
    # maximum of 7 TAF/day around mid-July
    D[t] = 5 + 2*np.sin(2*np.pi*dowy[t]/365 - np.pi)

    # old way:
    # if 120 < doy[t] < 270:
    #   D[t] = 7 # TAF/day
    # else:
    #   D[t] = 3

    # release and mass balance
    W = S[t-1] + Q[t]
    R[t] = min(P.evaluate([S[t-1], Q[t], dowy[t]]), W)
    spill[t] = max(W - R[t] - K, 0)
    S[t] = W - R[t] - spill[t]

    # pick one: either minimize deficit, or fit historical
    cost += max(D[t] - R[t], 0)**2

    # need to penalize large floods
    if R[t] + spill[t] > 228:
      cost += 10**10

    # ssq += (R[t] + spill - true_release[t])**2

  if mode is 'simulation':
    df['Ss'] = pd.Series(S, index=df.index)
    df['Rs'] = pd.Series(R, index=df.index)
    df['demand'] = pd.Series(D, index=df.index)
    df['spill'] = pd.Series(spill, index=df.index)
    return df
  else:
    return cost #deficit # minimize. If fitting historical.


def plot_results(df):
  plt.subplot(2,1,1)
  df.Ss.plot(color='blue', linewidth=2)
  df.storage.plot(color='cyan', linewidth=2)
  
  plt.subplot(2,1,2)
  (df.Rs + df.spill).plot(color='red', linewidth=2)
  df.outflow.plot(color='pink', linewidth=2)
  df.demand.plot(color='green', linewidth=2)
  plt.show()

# 2 features: water available, and month
feature_bounds = [[0,1000], [0,300], [1,365]]
names = ['Storage', 'Inflow', 'DOWY']
action_bounds = [0, 100] # max of 115000 cfs

algorithm = PTreeOpt(f, feature_bounds, action_bounds, feature_names=names)

# print algorithm.objectives

algorithm.run(max_nfe = 2000, log_frequency = 100)
print algorithm.best_P
print algorithm.best_f
# print [i for i in algorithm.population[0].L]
# f(algorithm.population[0], plot=True)

results = f(algorithm.best_P, mode='simulation')
plot_results(results)

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





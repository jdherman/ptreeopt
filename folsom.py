import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from opt import *

# simplified reservoir simulation

df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1995-10-01':'2015-09-30']

def water_day(d):
  return d - 274 if d >= 274 else d + 91

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def max_release(S):
  # rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
  storage = [0, 100, 400, 600, 1000]
  release = cfs_to_taf(np.array([0, 35000, 40000, 115000, 130000]))
  return np.interp(S, storage, release)


def f(P, mode='optimization'):

  # Set some parameters
  K = 975 # capacity, TAF

  # data setup
  Q = df.inflow.values
  true_release = df.outflow.values
  dowy = np.array([water_day(d) for d in df.index.dayofyear])
  T = len(Q)

  S,R,D = [np.zeros(T) for _ in range(3)]

  cost = 0
  ssq = 0

  S[0] = df.storage.values[0]

  for t in range(1,T):

    # demand: hypothetical model
    # minimum of 3 TAF/day around January 9th
    # maximum of 7 TAF/day around mid-July
    D[t] = 5 + 2*np.sin(2*np.pi*dowy[t]/365 - np.pi)

    # the real-valued way:
    # R[t] = min(P.evaluate([S[t-1], Q[t], dowy[t]]), W)

    TDI = Q[t:t+3].sum()

    # the discrete way:
    policy = P.evaluate([S[t-1], TDI])

    if policy == 'Release_Demand':
      target = D[t]
    elif policy == 'Hedge':
      target = 0.8*D[t]
    elif policy == 'Flood_Control':
      target = min(TDI, max_release(S[t-1]))

    # ramping rates
    p = 0.2
    R[t] = np.clip(R[t], (1-p)*R[t-1], (1+p)*R[t-1])

    R[t] = min(target, S[t-1] + Q[t]) # can't release more than available
    spill = max(S[t-1] + Q[t] - R[t] - K, 0)
    R[t] +=  spill
    S[t] = S[t-1] + Q[t] - R[t]

    # pick one: either minimize deficit, or fit historical
    cost += max(D[t] - R[t], 0)**2

    # need to penalize large floods (115k CFS maximum)
    # do this as just any overtopping, because it can't be controlled anymore
    # also penalize low storage volumes throughout.
    if R[t] > 220: # or spill:
      cost += 10**10
    # if S[t] < 200:
    #   cost += 10**5

    # ssq += (R[t] + spill - true_release[t])**2

  if mode is 'simulation':
    df['Ss'] = pd.Series(S, index=df.index)
    df['Rs'] = pd.Series(R, index=df.index)
    df['demand'] = pd.Series(D, index=df.index)
    return df
  else:
    return cost #deficit # minimize. If fitting historical.


def plot_results(df):
  plt.subplot(2,1,1)
  df.Ss.plot(color='blue', linewidth=2)
  df.storage.plot(color='cyan', linewidth=2)
  
  plt.subplot(2,1,2)
  df.Rs.plot(color='red', linewidth=2)
  df.outflow.plot(color='pink', linewidth=2)
  df.demand.plot(color='green', linewidth=2)
  plt.show()

# action_bounds = [0, 100] # max of 115000 cfs

algorithm = PTreeOpt(f, 
                    feature_bounds = [[0,1000], [0,500]],
                    feature_names = ['Storage', 'TDI'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge', 'Flood_Control']
                    )

# print algorithm.objectives

algorithm.run(max_nfe = 2000, log_frequency = 100)
print algorithm.best_P
print algorithm.best_f
# print [i for i in algorithm.population[0].L]
# f(algorithm.population[0], plot=True)

algorithm.best_P.graphviz_export('graphviz/bestPfol')

results = f(algorithm.best_P, mode='simulation')
plot_results(results)




# Need to add:


# feature variable names
# action variable name
# to make these work, they must be properties of the nodes

# discrete vs. continuous actions (it matters) -- not right now.
# operators (!!) mutation first

# to validate a tree ... this will be important
# lower nodes/logic (should not) contradict parent logic
# or else some branches will always evaluate false (this might be ok)





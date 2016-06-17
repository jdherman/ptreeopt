import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from opt import *

def water_day(d):
  return d - 274 if d >= 274 else d + 91

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def max_release(S):
  # rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
  storage = [0, 100, 400, 600, 1000]
  release = cfs_to_taf(np.array([0, 35000, 40000, 115000, 115000])) # make the last one 130 for future runs
  return np.interp(S, storage, release)


# simplified reservoir simulation
df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1955-10-01':'2015-09-30']
Q = df.inflow.values
K = 975 # capacity, TAF
dowy = np.array([water_day(d) for d in df.index.dayofyear])
D = 5 + 2*np.sin(2*np.pi*dowy/365 - np.pi)
T = len(Q)


def f(P, mode='optimization'):

  S,R = np.zeros(T),np.zeros(T)
  cost = 0
  S[0] = df.storage.values[0]

  for t in range(1,T):

    # if real-valued policy:
    # R[t] = min(P.evaluate([S[t-1], Q[t], dowy[t]]), W)
    TDI = np.sum(Q[t+1:t+4])

    # the discrete way:
    policy = P.evaluate([S[t-1], dowy[t], TDI])

    if policy == 'Release_Demand':
      target = D[t]
    elif policy == 'Hedge_80':
      target = 0.8*D[t]
    elif policy == 'Hedge_50':
      target = 0.5*D[t]
    elif policy == 'Flood_Control':
      # for optimization, use this. For fitting historical it won't work.
      target = max(S[t-1]+TDI-K,0)

    # a problem: flood control policy contributes to the demand objective
    # ie you can "satisfy demand" by releasing flood water.

    # ramping rates. need to clean up the logic of this part.
    # p = 0.2
    # R[t] = np.clip(R[t], (1-p)*R[t-1], (1+p)*R[t-1])

    # max/min release
    R[t] = np.clip(R[t], S[t-1] + Q[t], max_release(S[t-1]))
    R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
    S[t] = S[t-1] + Q[t] - R[t]

    # squared deficit. Also penalize any total release over 100 TAF/day  
    # should be able to vectorize this.  
    cost += max(D[t] - R[t], 0)**2 + max(R[t]-100, 0)**2


  if mode == 'simulation':
    df['Ss'] = pd.Series(S, index=df.index)
    df['Rs'] = pd.Series(R, index=df.index)
    df['demand'] = pd.Series(D, index=df.index)
    return df
  else:
    return cost #deficit # minimize. 
    # or if fitting historical ... (rmse)
    # return np.sqrt(np.mean((R - df.outflow.values)**2))


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
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'TDI'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_80', 'Hedge_50', 'Flood_Control']
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





import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
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

def tocs(d):
  # d must be water-year date
  # TAF of flood capacity in upstream reservoirs. simplified version.
  # approximate values of the curve here:
  # http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
  tp = [0, 50, 151, 200, 243, 366]
  sp = [975, 400, 400, 750, 975, 975]
  return np.interp(d, tp, sp)

# simplified reservoir simulation
df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1995-10-01':'2015-09-30']
Q = df.inflow.values
K = 975 # capacity, TAF
dowy = np.array([water_day(d) for d in df.index.dayofyear])
D = 4 + 3*np.sin(2*np.pi*dowy/365 - np.pi)
T = len(Q)

fit_historical = False

def f(P, mode='optimization'):

  S,R,target = np.zeros(T),np.zeros(T),np.zeros(T)
  cost = 0
  S[0] = df.storage.values[0]

  for t in range(1,T):

    TDI = np.sum(Q[t+1:t+4])
    policy = P.evaluate([S[t-1], dowy[t], TDI])

    if policy == 'Release_Demand':
      target[t] = D[t]
    elif policy == 'Hedge_80':
      target[t] = 0.8*D[t]
    elif policy == 'Hedge_50':
      target[t] = 0.5*D[t]
    elif policy == 'Flood_Control':
      if fit_historical:
        target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), 0.5*D[t])
      else:
        target[t] = max(S[t-1] + TDI - K, 0)

    # max/min release
    R[t] = min(target[t], S[t-1] + Q[t])
    R[t] = min(R[t], max_release(S[t-1]))
    R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
    S[t] = S[t-1] + Q[t] - R[t]

    # squared deficit. Also penalize any total release over 100 TAF/day  
    # should be able to vectorize this.  
    cost += max(D[t] - R[t], 0)**2 + max(R[t]-100, 0)**2


  if mode == 'simulation':
    df['Ss'] = pd.Series(S, index=df.index)
    df['Rs'] = pd.Series(R, index=df.index)
    df['demand'] = pd.Series(D, index=df.index)
    df['target'] = pd.Series(target, index=df.index)
    return df
  else:
    if fit_historical:
      return np.sqrt(np.mean((S - df.storage.values)**2))
    else:
      return cost


def plot_results(df):
  plt.subplot(2,1,1)
  df.Ss.plot(color='blue', linewidth=2)
  df.storage.plot(color='cyan', linewidth=2)
  
  plt.subplot(2,1,2)
  df.Rs.plot(color='red', linewidth=2)
  df.outflow.plot(color='pink', linewidth=2)
  df.demand.plot(color='green', linewidth=2)
  plt.show()

np.random.seed(13)


algorithm = PTreeOpt(f, 
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'TDI'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_80', 'Hedge_50', 'Flood_Control'],
                    mu = 7,
                    cx_prob = 0.50,
                    population_size = 50,
                    max_depth = 3
                    )


algorithm.run(max_nfe = 2000, log_frequency = 50, image_path = 'figs/anim/ptree')

print algorithm.best_P
print algorithm.best_f

algorithm.best_P.graphviz_export('figs/bestPfol')

results = f(algorithm.best_P, mode='simulation')
plot_results(results)


# TEST ONE
# L = [['Flood_Control']]
# L = [[1,220], ['Flood_Control'], ['Release_Demand']]
# P = PTree(L)
# P.graphviz_export('graphviz/whatever.png')
# results = f(P, mode='simulation')
# plot_results(results)
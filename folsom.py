from __future__ import division
import numpy as np 
import pandas as pd

def water_day(d):
  return d - 274 if d >= 274 else d + 91

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

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


class Folsom():

  def __init__(self, datafile, sd, ed, 
               cc = False, fit_historical = False):

    self.df = pd.read_csv(datafile, index_col=0, parse_dates=True)[sd:ed]
    self.K = 975 # capacity, TAF
    self.dowy = np.array([water_day(d) for d in self.df.index.dayofyear])
    self.D = np.loadtxt('demand.txt')[self.dowy]
    self.T = len(self.df.index)
    self.fit_historical = fit_historical


  def f(self, P, mode='optimization'):

    T = self.T
    S,R,target = [np.zeros(T) for _ in range(3)]
    cost = 0
    K = 975
    D = self.D
    Q = self.df.inflow.values
    dowy = self.dowy
    S[0] = self.df.storage.values[0]
    R[0] = D[0]
    policies = [None]

    for t in range(1,T):

      # TDI = np.sum(Q[t+1:t+4])
      policy = P.evaluate([S[t-1], self.dowy[t]])#, Q[t]]) # no inflow or forecast

      if policy == 'Release_Demand':
        target[t] = D[t]
      elif policy == 'Hedge_90':
        target[t] = 0.9*D[t]
      elif policy == 'Hedge_80':
        target[t] = 0.8*D[t]
      elif policy == 'Hedge_70':
        target[t] = 0.7*D[t]
      elif policy == 'Hedge_80':
        target[t] = 0.6*D[t]
      elif policy == 'Hedge_50':
        target[t] = 0.5*D[t]
      elif policy == 'Flood_Control':
        if self.fit_historical:
          target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), target[t])
        else:
          target[t] = max(0.2*(Q[t] + S[t-1] - 400), target[t])

      # old way ...
      # if flood_pool:
      #   target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), target[t])
      # elif policy == 'Flood_Control':
      #   target[t] = max_release(S[t-1]) # max(S[t-1] + Q[t] - K, 0)

      if mode == 'simulation':
        policies.append(policy)

      # max/min release
      R[t] = min(target[t], S[t-1] + Q[t])
      R[t] = min(R[t], max_release(S[t-1]))
      R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
      S[t] = S[t-1] + Q[t] - R[t]

      # squared deficit. Also penalize any total release over 100 TAF/day  
      # should be able to vectorize this.  
      cost += max(D[t] - R[t], 0)**2/T #+ max(R[t]-100, 0)**2

      if R[t] > cfs_to_taf(150000):
        cost += 10**8 # flood penalty, high enough to be a constraint


    if mode == 'simulation':
      df = self.df.copy()
      df['Ss'] = pd.Series(S, index=df.index)
      df['Rs'] = pd.Series(R, index=df.index)
      df['demand'] = pd.Series(D, index=df.index)
      df['target'] = pd.Series(target, index=df.index)
      df['policy'] = pd.Series(policies, index=df.index, dtype='category')
      return df
    else:
      if self.fit_historical:
        return np.sqrt(np.mean((S - self.df.storage.values)**2))
      else:
        return cost

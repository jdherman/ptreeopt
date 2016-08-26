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


def f(P, mode='optimization'):

  for s in scenarios:
    Q = inflows[s].values
    annQ = annQs[s].values
    lp3 = lp3s[s].values
    wycent = wycents[s].values

    S,R,target = np.zeros(T),np.zeros(T),np.zeros(T)
    cost = 0
    S[0] = 500 # assume. don't have df.storage.values[0]

    for t in range(1,T):

      # TDI = np.sum(Q[t+1:t+4])
      y = years[t]-2000
      policy = P.evaluate([S[t-1], dowy[t], annQ[y], lp3[y], wycent[y]])

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
        target[t] = max(0.2*(Q[t] + S[t-1] - 400), target[t])

      # if flood_pool:
      #   target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), target[t])
      # elif policy == 'Flood_Control':
      #   target[t] = max_release(S[t-1]) # max(S[t-1] + Q[t] - K, 0)

      # max/min release
      R[t] = min(target[t], S[t-1] + Q[t])
      R[t] = min(R[t], max_release(S[t-1]))
      R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
      S[t] = S[t-1] + Q[t] - R[t]

      # squared deficit. Also penalize any total release over 100 TAF/day  
      # should be able to vectorize this.  
      cost += max(D[t] - R[t], 0)**2/T #+ max(R[t]-100, 0)**2

      if R[t] > cfs_to_taf(130000):
        cost += 10**8 # flood penalty, high enough to be a constraint


  return cost / len(scenarios)


# run this stuff whenever the file is imported
inflows = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0, parse_dates=True)
inflows = inflows['2000':]
scenarios = [s for s in inflows.columns if 'canesm2_rcp85_r1i1p1' in s] # Only running rcp8.5 right now
years = inflows.index.year
dowy = np.array([water_day(d) for d in inflows.index.dayofyear])
D = np.loadtxt('demand.txt')[dowy]
annQs = pd.read_csv('data/folsom-cc-annQ-MA50.csv', index_col=0, parse_dates=True)
lp3s = pd.read_csv('data/folsom-cc-lp3-kcfs.csv', index_col=0, parse_dates=True)
wycents = pd.read_csv('data/folsom-cc-wycentroid.csv', index_col=0, parse_dates=True)
K = 975 # capacity, TAF
T = len(inflows.index)

flood_pool = False

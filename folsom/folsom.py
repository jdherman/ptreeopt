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
  storage = [90, 100, 400, 600, 975]
  release = cfs_to_taf(np.array([0, 35000, 40000, 115000, 130000])) # make the last one 130 for future runs
  return np.interp(S, storage, release)

def tocs(d):
  # d must be water-year date
  # TAF of flood capacity in upstream reservoirs. simplified version.
  # approximate values of the curve here:
  # http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
  tp = [0, 50, 151, 200, 243, 366]
  sp = [975, 400, 400, 750, 975, 975]
  return np.interp(d, tp, sp)

def volume_to_height(S): # from HOBBES data
  sp = [0, 48, 93, 142, 192, 240, 288, 386, 678, 977]
  ep = [210, 305, 332, 351, 365, 376, 385, 401, 437, 466]
  return np.interp(S, sp, ep)


class Folsom():

  def __init__(self, datafile, sd, ed,
               fit_historical = False, use_tocs = False, 
               cc = False, scenario = None, multiobj = False):

    self.df = pd.read_csv(datafile, index_col=0, parse_dates=True)[sd:ed]
    self.K = 975 # capacity, TAF
    self.turbine_elev = 134 # feet
    self.turbine_max_release = 8600 # cfs
    self.max_safe_release = 130000 # cfs

    self.dowy = np.array([water_day(d) for d in self.df.index.dayofyear])
    self.D = np.loadtxt('folsom/data/demand.txt')[self.dowy]
    self.T = len(self.df.index)
    self.fit_historical = fit_historical
    self.use_tocs = use_tocs
    self.cc = cc
    self.multiobj = multiobj


    if self.cc:
      self.annQs = pd.read_csv('folsom/data/folsom-cc-annQ-MA30.csv', index_col=0, parse_dates=True)
      self.lp3s = pd.read_csv('folsom/data/folsom-cc-lp3-kcfs.csv', index_col=0, parse_dates=True)
      self.wycs = pd.read_csv('folsom/data/folsom-cc-wycentroid.csv', index_col=0, parse_dates=True)
      self.years = self.df.index.year
      if scenario:
        self.set_scenario(scenario)
    else:
      self.Q = self.df.inflow.values


  def set_scenario(self, s):
    self.scenario = s
    self.annQ = self.annQs[s].values
    self.lp3 = self.lp3s[s].values
    self.wyc = self.wycs[s].values
    self.Q = self.df[s].values


  def f(self, P, mode='optimization'):

    T = self.T
    S,R,target,shortage_cost,flood_cost = [np.zeros(T) for _ in range(5)]
    K = self.K
    D = self.D
    Q = self.Q
    dowy = self.dowy
    R[0] = D[0]
    policies = [None]

    if not self.cc:
      S[0] = self.df.storage.values[0]
    else:
      S[0] = 500

    for t in range(1,T):

      if not self.cc:
        policy,rules = P.evaluate([S[t-1], self.dowy[t], Q[t]])
      else:
        y = self.years[t]-2000
        policy,rules = P.evaluate([S[t-1], Q[t], dowy[t], 
                                  self.annQ[y], self.lp3[y], self.wyc[y]])
      
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

      if self.use_tocs:
        target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), target[t])
      elif policy == 'Flood_Control':
        # target[t] = max_release(S[t-1])        
        target[t] = max(0.2*(Q[t] + S[t-1] - 0.0), 0.0) # default
        # for item in rules:
        #   if item[0] == 'Storage' and not item[2]:
        #     target[t] = max(0.2*(Q[t] + S[t-1] - item[1]), 0.0)

      if mode == 'simulation':
        policies.append(policy)

      # max/min release
      # k = 0.2
      R[t] = min(target[t], S[t-1] + Q[t])
      R[t] = min(R[t], max_release(S[t-1]))
      # R[t] = np.clip(R[t], (1-k)*R[t], (1+k)*R[t]) # inertia -- 
      R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
      S[t] = S[t-1] + Q[t] - R[t]

      # squared deficit. Also penalize any total release over 100 TAF/day  
      # should be able to vectorize this.  
      shortage_cost[t] = max(D[t] - R[t], 0)**2/T #+ max(R[t]-100, 0)**2

      if R[t] > cfs_to_taf(self.max_safe_release):
        flood_cost[t] += 10**3 * (R[t] - cfs_to_taf(self.max_safe_release)) # flood penalty, high enough to be a constraint

    # end of period penalty
    # EOP = 0
    # if not self.cc and S[-1] < self.df.storage.values[0]:
    #   EOP = 10**5 

    if mode == 'simulation' or self.multiobj:
      df = self.df.copy()
      df['Ss'] = pd.Series(S, index=df.index)
      df['Rs'] = pd.Series(R, index=df.index)
      df['demand'] = pd.Series(D, index=df.index)
      df['target'] = pd.Series(target, index=df.index)
      head = (volume_to_height(df.Ss) - self.turbine_elev)
      power_release = taf_to_cfs(df.Rs.copy()).clip(0, self.turbine_max_release)
      df['power'] = (24*0.85*10**-4/1.181)*(head*power_release)


    if mode == 'simulation':
      df['policy'] = pd.Series(policies, index=df.index, dtype='category')
      return df    
    else:
      if self.fit_historical:
        return np.sqrt(np.mean((S - self.df.storage.values)**2))
      else:
        if not self.multiobj:
          return shortage_cost.sum() + flood_cost.sum() # + EOP
        else:
          J1 = shortage_cost.sum() # water supply
          J2 = taf_to_cfs(df.Rs.max()) # peak flood
          # (3) environmental alteration:
          # integrate between inflow/outflow exceedance curves
          ixi = np.argsort(Q)
          ixo = np.argsort(R)
          J3 = np.trapz(np.abs(Q[ixi]-R[ixo]), dx=1.0/T)
          # (4) Maximize hydropower generation (GWh)
          # Max turbine release 8600 cfs, 215 MW. Average annual production 620 GWh.
          # http://www.usbr.gov/mp/EWA/docs/DraftEIS-Vol2/Ch16.pdf
          # http://rivers.bee.oregonstate.edu/book/export/html/6
          J4 = -df.power.resample('AS-OCT').sum().mean() / 1000 
          return [J1,J2,J3,J4]



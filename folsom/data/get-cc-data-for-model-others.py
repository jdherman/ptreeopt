from __future__ import division
import numpy as np 
import pandas as pd
from pandas import *
from scipy.stats import norm,skew

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def bias_correct(Q):
  return Q * 2700.0/3500.0 # because CMIP doesn't match historical

dff = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                  parse_dates={'datetime': [0,1,2]},
                  date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))


##################################
# Annual inflow

cmip_annQ_MA = (dff.pipe(cfs_to_taf)
                   .pipe(bias_correct)
                   .resample('AS-OCT').sum()
                   .rolling(window=10)
                   .mean())['2000':]

###################################
# Max inflow
dmax = 5
cmip_dmax = (dff.pipe(cfs_to_taf)
                   .rolling(dmax).sum()
                   .resample('AS-OCT')
                   .max())['2000':]

###################################
# LP3 flood

def lp3(Q, ppf=0.99):
  n = len(Q)
  Q = np.log(Q)
  m = Q.mean()
  s = Q.std()
  g = skew(Q)

  # Log-space moment estimators, HH 18.2.27
  ahat = 4/g**2
  bhat = 2/(s*g)
  zhat = m - 2*s/g
  # Frequency factor Kp, HH 18.2.29
  Kp = (2/g)*(1 + g*norm.ppf(ppf)/6 - g**2/36)**3 - 2/g
  q99 = np.exp(m + s*Kp)
  return q99


cmip_lp3 = (dff.resample('AS-OCT')
               .max()
               .expanding(min_periods=50) # instead of rolling
               .apply(lp3)/1000)['2000':]


####################################
# season shift
def water_day(d):
    doy = d.dayofyear
    return doy - 274 if doy >= 274 else doy + 91

def get_wday(s):
  total = s.sum()
  cs = s.cumsum()
  day = s.index[cs > 0.5*total][0]
  return water_day(day)


cmip_dowy = (dff.pipe(cfs_to_taf)
                .resample('AS-OCT')
                .apply(get_wday)
                .rolling(window=50)
                .mean())['2000':]

### save everything

# dff.pipe(cfs_to_taf)['2000':].to_csv('folsom-cc-inflows.csv')

cmip_annQ_MA.to_csv('folsom-cc-annQ-MA10.csv')

# cmip_dmax.to_csv('folsom-cc-%ddmax.csv' % dmax)

# cmip_lp3.to_csv('folsom-cc-lp3-kcfs.csv')

# cmip_dowy.to_csv('folsom-cc-wycentroid.csv')


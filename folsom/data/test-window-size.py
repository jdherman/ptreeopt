from __future__ import division
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def bias_correct(Q):
  return Q * 2700.0/3500.0 # because CMIP doesn't match historical

dff = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                  parse_dates={'datetime': [0,1,2]},
                  date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))


# dfh = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)
# annsum = dfh.resample('AS-OCT').sum()

annsum = (dff.pipe(cfs_to_taf)
             .pipe(bias_correct)
             .resample('AS-OCT').sum())

# past = annsum.rolling(15).mean()
# ratio = (past / past.values[15,:])
# ratio.plot(color='indianred', linewidth=0.5, legend=None)
# plt.show()


# THIS IS NOT GOOD

fw = 1

for w in [1,3,5,10,30,50]:

  past = annsum.rolling(window=w).mean()
  future = annsum.rolling(window=fw).mean()

  x = past.values[(w-1):(-fw),:].flatten()
  y = future.values[(w-1+fw):,:].flatten()
  _,_,r,_,_ = linregress(x,y)
  print '%d-year MA: %f' % (w,r)


# for c in annsum.columns:
#   print annsum[c].autocorr(lag=1)




from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

df = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                                                        parse_dates={'datetime': [0,1,2]},
                                                        date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))

# FNF = pd.read_csv('dwr-fnf.txt', delim_whitespace=True, index_col=0, parse_dates=True)

# df2 = pd.read_csv('../model-data.csv', index_col=0, parse_dates=True).resample('A', how='sum')

# df.resample('A', how='sum').autocorr(lag=1).plot(legend=None)
# USE HISTORICAL DATA, again. CC predictions aren't good.
pd.rolling_std(df, window=10).plot(legend=None)
pd.rolling_mean(df, window=10).plot(legend=None)
plt.show()
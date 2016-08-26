from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
from scipy import stats

def init_plotting():
  sns.set_style('whitegrid')

  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (12, 4)
  plt.rcParams['font.size'] = 13
  plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def bias_correct(Q):
  return Q * 2700.0/3500.0 # because CMIP doesn't match historical

dfh = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)
# df = df['1995-10-01':'2015-09-30']
dff = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                  parse_dates={'datetime': [0,1,2]},
                  date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))

init_plotting()

k = dff.columns[24]

hist = np.log10(dfh['1950':'2000'].inflow)
cmip5 = np.log10(cfs_to_taf(dff)['1950':'2000'])
dff = 10**(((np.log10(cfs_to_taf(dff)) - cmip5.mean())/cmip5.std())*hist.std() + hist.mean())

# sns.distplot(hist, kde=False, fit=stats.norm, bins=30, color='steelblue')
# sns.distplot(dff[k], kde=False, fit=stats.norm, bins=30, color='indianred')

ax = (10**hist).plot()
(10**cmip5[k]).plot(ax=ax)

plt.tight_layout()
plt.show()

# dff.to_csv('cmip5-folsom-qq.csv')
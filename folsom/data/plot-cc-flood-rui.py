from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
from scipy.stats import skew, norm

def init_plotting():
  sns.set_style('whitegrid')

  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (9, 7)
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

dfh = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)

dff = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                  parse_dates={'datetime': [0,1,2]},
                  date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))

init_plotting()

rcpcolors = ['#FFD025', '#E87F19', '#FF4932']
rcps = ['rcp45', 'rcp60', 'rcp85']

ax = plt.subplot(1,2,1)


hist_annmax = (dfh.inflow.pipe(taf_to_cfs)
               .resample('AS-OCT')
               .max()/1000) # kcfs

hist_mu =  (hist_annmax
               .rolling(50) # instead of rolling
               .mean())

cmip_mu = (dff.resample('AS-OCT')
               .max()
               .rolling(50) # instead of rolling
               .mean()/1000)['2016':]

for r,c in zip(rcps, rcpcolors):
  (cmip_mu.filter(regex=r)
          .plot(legend=None, color=c, ax=ax, linewidth=0.5))

# hist_annmax.plot(legend=None, ls='None', marker='^', mfc='k', ax=ax, markersize=3)
hist_mu.plot(legend=None, color='k', ax=ax, linewidth=2)

plt.xlabel('')
plt.ylabel('Inflow, mean daily kcfs')
plt.title('(a) Mean Annual Peak Flow (50-yr MA)', family='OfficinaSanITCMedium', loc='left')
plt.xlim(['1955','2100'])
plt.ylim(0,150)
plt.axvline('2015',ymin=0,ymax=1, color='k')
plt.annotate('HISTORICAL', xy=('1973',130), family='OfficinaSanITCBooIta', color='k')
plt.annotate('CMIP5', xy=('2021', 130), family='OfficinaSanITCBooIta', color='k')
# plt.annotate('RCP4.5', xy=('2021',70), family='OfficinaSanITCMedium', color=rcpcolors[0])
# plt.annotate('RCP6.0', xy=('2021',40), family='OfficinaSanITCMedium', color=rcpcolors[1])
# plt.annotate('RCP8.5', xy=('2021',10), family='OfficinaSanITCMedium', color=rcpcolors[2])


##############################
ax = plt.subplot(1,2,2)

hist_std =  (hist_annmax
               .rolling(50) # instead of rolling
               .std())

cmip_std = (dff.resample('AS-OCT')
               .max()
               .rolling(50) # instead of rolling
               .std()/1000)['2016':]

for r,c in zip(rcps, rcpcolors):
  (cmip_std.filter(regex=r)
          .plot(legend=None, color=c, ax=ax, linewidth=0.5))

# hist_annmax.plot(legend=None, ls='None', marker='^', mfc='k', ax=ax, markersize=3)
hist_std.plot(legend=None, color='k', ax=ax, linewidth=2)

plt.xlabel('')
# plt.ylabel('Inflow, mean daily kcfs')
plt.title('(b) Stdev Annual Peak Flow (50-yr MA)', family='OfficinaSanITCMedium', loc='left')
plt.xlim(['1955','2100'])
plt.ylim(0,150)
plt.axvline('2015',ymin=0,ymax=1, color='k')
plt.annotate('HISTORICAL', xy=('1973',130), family='OfficinaSanITCBooIta', color='k')
plt.annotate('CMIP5', xy=('2021', 130), family='OfficinaSanITCBooIta', color='k')
# plt.annotate('RCP4.5', xy=('2021',70), family='OfficinaSanITCMedium', color=rcpcolors[0])
# plt.annotate('RCP6.0', xy=('2021',40), family='OfficinaSanITCMedium', color=rcpcolors[1])
# plt.annotate('RCP8.5', xy=('2021',10), family='OfficinaSanITCMedium', color=rcpcolors[2])



plt.show()

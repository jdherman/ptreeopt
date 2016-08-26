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

dfh = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)

dff = pd.read_csv('streamflow_cmip5_ncar_day_FOL_I.csv', index_col='datetime', 
                  parse_dates={'datetime': [0,1,2]},
                  date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))

init_plotting()

rcpcolors = ['#FFD025', '#E87F19', '#FF4932']
rcps = ['rcp45', 'rcp60', 'rcp85']

##################################
# Annual inflow

cmip_annQ_MA = (dff.pipe(cfs_to_taf)
                   .pipe(bias_correct)
                   .resample('AS-OCT').sum()
                   .rolling(window=50)
                   .mean())['2016':]

hist_annQ_MA = (dfh.resample('AS-OCT').sum()
                   .rolling(window=50)
                   .mean())

ax = plt.subplot(2,2,1)

for r,c in zip(rcps, rcpcolors):
  (cmip_annQ_MA.filter(regex=r)
               .plot(legend=None, color=c, ax=ax, linewidth=0.5))

hist_annQ_MA.inflow.plot(legend=None, color='k', ax=ax, linewidth=2)

plt.xlabel('')
plt.ylabel('Inflow, TAF/year')
plt.title('(a) Annual Inflow, 50-year MA', family='OfficinaSanITCMedium', loc='left')
plt.xlim(['1955','2100'])
plt.axvline('2015',ymin=0,ymax=1, color='k')
plt.annotate('HISTORICAL', xy=('1973',4200), family='OfficinaSanITCBooIta', color='k')
plt.annotate('CMIP5', xy=('2021',4200), family='OfficinaSanITCBooIta', color='k')
plt.annotate('RCP4.5', xy=('2021',1700), family='OfficinaSanITCMedium', color=rcpcolors[0])
plt.annotate('RCP6.0', xy=('2021',1400), family='OfficinaSanITCMedium', color=rcpcolors[1])
plt.annotate('RCP8.5', xy=('2021',1100), family='OfficinaSanITCMedium', color=rcpcolors[2])

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


ax = plt.subplot(2,2,2)


hist_annmax = (dfh.inflow.pipe(taf_to_cfs)
               .resample('AS-OCT')
               .max()/1000) # kcfs

hist_lp3 =  (hist_annmax
               .expanding(min_periods=50) # instead of rolling
               .apply(lp3))

cmip_lp3 = (dff.resample('AS-OCT')
               .max()
               .expanding(min_periods=50) # instead of rolling
               .apply(lp3)/1000)['2016':]

for r,c in zip(rcps, rcpcolors):
  (cmip_lp3.filter(regex=r)
          .plot(legend=None, color=c, ax=ax, linewidth=0.5))

# hist_annmax.plot(legend=None, ls='None', marker='^', mfc='k', ax=ax, markersize=3)
hist_lp3.plot(legend=None, color='k', ax=ax, linewidth=2)

plt.xlabel('')
plt.ylabel('Inflow, mean daily kcfs')
plt.title('(b) LP3 100-year flood estimate', family='OfficinaSanITCMedium', loc='left')
plt.xlim(['1955','2100'])
plt.ylim(0,350)
plt.axvline('2015',ymin=0,ymax=1, color='k')
plt.annotate('HISTORICAL', xy=('1973',320), family='OfficinaSanITCBooIta', color='k')
plt.annotate('CMIP5', xy=('2021', 320), family='OfficinaSanITCBooIta', color='k')
plt.annotate('RCP4.5', xy=('2021',70), family='OfficinaSanITCMedium', color=rcpcolors[0])
plt.annotate('RCP6.0', xy=('2021',40), family='OfficinaSanITCMedium', color=rcpcolors[1])
plt.annotate('RCP8.5', xy=('2021',10), family='OfficinaSanITCMedium', color=rcpcolors[2])

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

ax = plt.subplot(2,2,3)

def wdaymean(x):
  return x.groupby(water_day).mean()

# moving average across 30 days to filter noise
ts = dfh.inflow.rolling(30, center=True).mean()

A = ts.resample('20AS-OCT').apply(wdaymean)
ct = 0
grays = ['#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']

for n,g in A.groupby(level=0):
  plt.plot(g.values, color=grays[ct], linewidth=2)
  ct += 1

legendstr = ['1904-1924', '1924-1944', '1944-1964', '1964-1984', '1984-2004', '2004-2015']
plt.xlim([0,365])
plt.xlabel('Day of Water Year')
plt.ylabel('Moving-average inflow, TAF')
plt.title('(c) Seasonal shift, historical', family='OfficinaSanITCMedium', loc='left')

for i,l in enumerate(legendstr):
  plt.annotate(l, xy=('277',18-2*i), family='OfficinaSanITCMedium', color=grays[i])



##############################################################
# water year centroid
ax = plt.subplot(2,2,4)

hist_dowy = (dfh.inflow
          .resample('AS-OCT')
          .apply(get_wday)
          .rolling(window=50)
          .mean())

cmip_dowy = (dff.pipe(cfs_to_taf)
                .resample('AS-OCT')
                .apply(get_wday)
                .rolling(window=50)
                .mean())['2016':]

for r,c in zip(rcps, rcpcolors):
  (cmip_dowy.filter(regex=r)
          .plot(legend=None, color=c, ax=ax, linewidth=0.5))

# cmip_dowy.plot(legend=None, color='steelblue', linewidth=0.5, ax=ax)

hist_dowy.plot(color='k', linewidth=2, ax=ax)

plt.xlabel('')
plt.ylabel('Day of Water Year')
plt.title('(d) Water year centroid, 50-year MA', family='OfficinaSanITCMedium', loc='left')
plt.xlim(['1955','2100'])
# plt.ylim(0,350)
plt.axvline('2015',ymin=0,ymax=1, color='k')
plt.annotate('HISTORICAL', xy=('1973',184), family='OfficinaSanITCBooIta', color='k')
plt.annotate('CMIP5', xy=('2021', 184), family='OfficinaSanITCBooIta', color='k')
plt.annotate('RCP4.5', xy=('2021',134), family='OfficinaSanITCMedium', color=rcpcolors[0])
plt.annotate('RCP6.0', xy=('2021',128), family='OfficinaSanITCMedium', color=rcpcolors[1])
plt.annotate('RCP8.5', xy=('2021',122), family='OfficinaSanITCMedium', color=rcpcolors[2])


###############################
plt.tight_layout()
# plt.show()
plt.savefig('combined.svg')
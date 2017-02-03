import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from datetime import timedelta


def init_plotting():
  sns.set_style('whitegrid')

  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (9,4)
  plt.rcParams['font.size'] = 13
  plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


init_plotting()


###########################################
plt.subplot(1,2,2)

df = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)
df = df['19951001':'20150930']

def water_day(d):
    doy = d.dayofyear
    if doy >= 274:
        outdate = doy - 274
    else:
        outdate = doy + 91
    return outdate

def flood_rule(d): # d must be water year day
  # assuming upstream storage of 200 TAF available
  # http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
  tp = [0, 50, 151, 200, 243, 366]
  sp = [975, 575, 575, 750, 975, 975]
  return np.interp(d, tp, sp)

df['water_day'] = pd.Series([water_day(d) for d in df.index], index=df.index)
df['flood_rule'] = pd.Series([flood_rule(d) for d in df.water_day], index=df.index)
df['frac'] = (df.storage+df.inflow) / df.flood_rule
df['median_outflow'] = pd.Series([df.outflow[(df.water_day==d) & (df.frac < 0.8) & (df.outflow < 11)].median() for d in df.water_day], index=df.index)
df['mean_outflow'] = pd.Series([df.outflow[(df.water_day==d) & (df.frac < 0.8) & (df.outflow < 11)].mean() for d in df.water_day], index=df.index)
df['std_outflow'] = pd.Series([df.outflow[(df.water_day==d) & (df.frac < 0.8) & (df.outflow < 11)].std() for d in df.water_day], index=df.index)

# df[(df.outflow < 11) & (df.frac < 0.8) & (df.storage > 0) & (df.storage < 1000)].plot(kind='scatter', x='water_day', y='outflow', c='steelblue', edgecolor='none')

# don't do this...
# df['demand'] = pd.Series([(4 + 2*np.sin(2*np.pi*d/365 - np.pi)) for d in df.water_day], index=df.index)

# do this instead
df['demand'] = df['median_outflow'].rolling(window=25, center=True).mean()

mu = df['median_outflow'][0:366].values
sig = df['std_outflow'][0:366].values
plt.fill_between(range(366), mu+sig, mu-sig, facecolor='steelblue', alpha=0.5, edgecolor='None')
plt.plot(mu, color='steelblue', linewidth=2)

plt.plot(df['demand'][732:1098].values, color='k', linewidth=2)

plt.xlim([0,365])
plt.ylim([0,10])
plt.xlabel('Day of Water Year')
plt.ylabel('Release (TAF/day)')
plt.title('Daily Releases, 1995-2015', family='OfficinaSanITCMedium')
plt.legend(['Historical $\mu \pm \sigma$', 'MA$_{25}$ (centered)'], loc='upper left')


np.savetxt('demand.txt', df['demand'][732:1098].values)

###############################################
plt.subplot(1,2,1)
df = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)
df = df['19551001':'20150930']


def annfdc(a):
  a['exceedance'] = a.inflow.rank(pct=True, ascending=False)
  a = a.sort_values(by='exceedance')
  plt.plot(a.exceedance.values, np.log10(a.inflow.values), color='steelblue')

df.resample('A').apply(annfdc)
# plt.plot(df.exceedance.values, np.log10(df.inflow.values))
plt.ylim([-1,3])
plt.ylabel('Inflow, $\log_{10}$ TAF/day')
plt.xlabel('Exceedance')
plt.title('Annual Exceedance Curves, 1955-2015', family='OfficinaSanITCMedium')


plt.tight_layout()
# plt.show()
plt.savefig('folsom-historical-stats.svg')

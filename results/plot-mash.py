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

df = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True)
# df = df['19951001':'20150930']

colors = ['#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594']

def water_day(d):
    doy = d.dayofyear
    if doy >= 274:
        outdate = doy - 274
    else:
        outdate = doy + 91
    return outdate

df['water_day'] = pd.Series([water_day(d) for d in df.index], index=df.index)
# df['mean_inflow'] = pd.Series([df.inflow[(df.water_day==d)].mean() for d in df.water_day], index=df.index)

def wdaymean(x):
  return x.groupby('water_day').mean()

# moving average across 20 days to filter noise
df.inflow = df.inflow.rolling(30, center=True).mean()

A = df.resample('20AS-OCT').apply(wdaymean).inflow
ct = 0

for n,g in A.groupby(level=0):
  print n.year
  plt.plot(g.values, color=colors[ct], linewidth=2)
  ct += 1

plt.legend(['1904-1924', '1924-1944', '1944-1964', '1964-1984', '1984-2004', '2004-2015'])
plt.xlim([0,365])
plt.xlabel('Day of Water Year')
plt.ylabel('Moving-average inflow, TAF')
plt.title('Moving Average over Shifting Horizon')
plt.show()


# A = pd.rolling_apply(df.inflow, 20*365, wdaymean, center=True)
# # print (df.inflow.rolling(20*365, center=True)
# # #   .apply(wdaymean))

# A = (df.groupby('water_day')
#        .inflow.mean())
       # .apply(pd.rolling_mean, 20, center=True))
       
#        # .resample('AS-OCT'))
# grouped = df.groupby('water_day').inflow

# for name, group in grouped:
#   group.rolling(20).mean().plot()

# plt.show()

# print A
# A.apply(pd.rolling_mean, 1).plot()
# plt.show()

# A = (df.inflow
#        # .resample('AS-OCT')
#        .rolling(20*365)
#        .apply(wdaymean))

# print A
# # print A.head()
# # A.mean().plot()
# # plt.show()

# df.groupby('water_day').inflow.mean().plot()
plt.show()
# g = df.groupby('water_day')
# # print g.describe()

# g.inflow.apply(pd.rolling_mean, window=20).plot()
# plt.show()
# # do this instead
# df['demand'] = df['mean_inflow'].rolling(window=25, center=True).mean()

# mu = df['median_outflow'][0:366].values
# plt.plot(mu, color='steelblue', linewidth=2)

# plt.plot(df['demand'][732:1098].values, color='k', linewidth=2)

# plt.xlim([0,365])
# # plt.ylim([0,10])
# plt.xlabel('Day of Water Year')
# # plt.ylabel('Release (TAF/day)')
# plt.title('Daily Releases, 1995-2015', family='OfficinaSanITCMedium')
# plt.legend(['Historical $\mu \pm \sigma$', 'MA$_{25}$ (centered)'], loc='upper left')




# plt.tight_layout()
# plt.show()
# plt.savefig('folsom-historical-stats.svg')

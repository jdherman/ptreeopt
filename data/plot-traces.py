import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def init_plotting(w,h):
  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (w,h)
  plt.rcParams['font.size'] = 13
  plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


init_plotting(5,4)


# inflows = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0, parse_dates=True)
# inflows = inflows['2050':]
# scenarios = [s for s in inflows.columns if 'rcp85' in s] # Only running rcp8.5 right now
# years = inflows.index.year
# dowy = np.array([water_day(d) for d in inflows.index.dayofyear])
# D = np.loadtxt('demand.txt')[dowy]
inflows = pd.read_csv('folsom-cc-inflows.csv', index_col=0, parse_dates=True)

# wet, dry, middle
names = ['canesm2_rcp85_r1i1p1', 'miroc-esm_rcp85_r1i1p1', 'gfdl-esm2g_rcp85_r1i1p1']
inflows[names].plot()
# for r,c in zip(rcps, rcpcolors):
#   x = annQs['2099'].filter(regex=r).values[0]
#   y = lp3s['2099'].filter(regex=r).values[0]
#   plt.scatter(x,y, s=50, color=c, alpha=0.5)
#   plt.hold(True)

# for i,name in enumerate(names):
#   x = annQs['2099'][name].values[0]
#   y = lp3s['2099'][name].values[0]
#   plt.annotate(name.split('_',1)[0], (x+40,y+3))

# cX,cY = (2700,225)
# plt.scatter(cX,cY,s=70,c='k')
# plt.annotate('Current', (cX+70,cY-4), family='OfficinaSanITCMedium')

# plt.annotate('RCP4.5', xy=(3800,170), family='OfficinaSanITCMedium', color=rcpcolors[0])
# plt.annotate('RCP6.0', xy=(3800,145), family='OfficinaSanITCMedium', color=rcpcolors[1])
# plt.annotate('RCP8.5', xy=(3800,120), family='OfficinaSanITCMedium', color=rcpcolors[2])



# plt.xlabel('Annual Inflow, 50-year MA (TAF/yr)')
# plt.ylabel('LP3 100-year flood estimate (TAF/d)')
# plt.tight_layout()
plt.show()
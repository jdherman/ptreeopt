import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


def init_plotting(w, h):
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (w, h)
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


init_plotting(6, 5)


# inflows = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0, parse_dates=True)
# inflows = inflows['2050':]
# scenarios = [s for s in inflows.columns if 'rcp85' in s] # Only running rcp8.5 right now
# years = inflows.index.year
# dowy = np.array([water_day(d) for d in inflows.index.dayofyear])
# D = np.loadtxt('demand.txt')[dowy]


# first just the scatter points

path = '../data/'
annQs = pd.read_csv(path + 'folsom-cc-annQ-MA30.csv',
                    index_col=0, parse_dates=True)
anndmax = pd.read_csv(path + 'folsom-cc-7dmax.csv',
                      index_col=0, parse_dates=True)
# lp3s = pd.read_csv(path + 'folsom-cc-lp3-kcfs.csv', index_col=0, parse_dates=True)
# wycents = pd.read_csv(path + 'folsom-cc-wycentroid.csv', index_col=0, parse_dates=True)

rcpcolors = ['#FFD025', '#E87F19', '#FF4932']
rcps = ['rcp45', 'rcp60', 'rcp85']
# wet, dry, middle
names = ['canesm2_rcp85_r1i1p1',
         'miroc-esm_rcp85_r1i1p1', 'gfdl-esm2g_rcp85_r1i1p1']

for r, c in zip(rcps, rcpcolors):
    x = annQs['2099'].filter(regex=r).values[0]
    y = anndmax.max(axis=0).filter(regex=r)
    plt.scatter(x, y, s=60, color=c, alpha=0.5)
    plt.hold(True)

# for i,name in enumerate(annQs.columns):
#   # x = annQs['2099'][name].values[0]
#   # y = lp3s['2099'][name].values[0]
#   x = annQs['2099'][name].values[0]
#   y = anndmax.max(axis=0)[name]
#   # print (name,x,y)
#   plt.annotate(name, (x+40,y+3)) # name.split('_',1)[0]


# plt.title('2100 Runoff Statistics')


# What is the current value for the "dmax" value?
# Max inflow
dmax = 7
dfh = pd.read_csv('../folsom-daily.csv', index_col=0, parse_dates=True).inflow
cY = (dfh.rolling(dmax).sum()
      .resample('AS-OCT')
      .max()
      .max(axis=0))
cX = (dfh.resample('AS-OCT').sum().rolling(30).mean()['2014'])
plt.scatter(cX, cY, s=70, c='k')
# plt.annotate('Current', (cX+70,cY-4), family='OfficinaSanITCMedium')

plt.annotate('RCP4.5', xy=(1550, 3750),
             family='OfficinaSanITCMedium', color=rcpcolors[0])
plt.annotate('RCP6.0', xy=(1550, 3550),
             family='OfficinaSanITCMedium', color=rcpcolors[1])
plt.annotate('RCP8.5', xy=(1550, 3350),
             family='OfficinaSanITCMedium', color=rcpcolors[2])


plt.xlabel('Mean Annual Inflow, 2070-2100 (TAF/yr)')
plt.ylabel('Max 7-day flow, 2000-2100 (TAF/7-day)')


# then include the optimization results

scenarios = pd.read_csv(path + 'folsom-cc-inflows.csv', index_col=0).columns

for s in scenarios:
    if 'rcp26' not in s:
        data = pickle.load(open('cc-opt/snapshots-cc-' + s + '.pkl', 'rb'))

        nfe = data['nfe']
        best_f = np.array(data['best_f'])
        best_P = data['best_P'][-1]
        elem = [str(i) for i in best_P.L]

        if (best_f[-1] > 10.0):  # ('Release_Demand' not in elem) or
            x = annQs['2099'][s].values[0]
            y = anndmax[s].max(axis=0)
            plt.scatter(x, y, marker='x', linewidth=1.5, color='k')

    # print '%s, %f, %s' % (s, best_f[-1], data['best_P'][-1]) # to see tree logic


# want to highlight just a few
names = ['access1-0_rcp45_r1i1p1',
         'canesm2_rcp85_r1i1p1',
         'giss-e2-h-cc_rcp45_r1i1p1',
         'gfdl-esm2g_rcp45_r1i1p1',
         'miroc-esm_rcp85_r1i1p1']
for s in names:
    x = annQs['2099'][s].values[0]
    y = anndmax[s].max(axis=0)
    plt.scatter(x, y, 100, marker='s', linewidth=1.5,
                color='k', facecolor='None')


plt.tight_layout()
# plt.show()
plt.savefig('cc-scatter.svg')

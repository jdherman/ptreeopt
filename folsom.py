import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def water_day(d):
  return d - 274 if d >= 274 else d + 91

def cfs_to_taf(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def max_release(S):
  # rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
  storage = [0, 100, 400, 600, 1000]
  release = cfs_to_taf(np.array([0, 35000, 40000, 115000, 115000])) # make the last one 130 for future runs
  return np.interp(S, storage, release)

def tocs(d):
  # d must be water-year date
  # TAF of flood capacity in upstream reservoirs. simplified version.
  # approximate values of the curve here:
  # http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
  tp = [0, 50, 151, 200, 243, 366]
  sp = [975, 400, 400, 750, 975, 975]
  return np.interp(d, tp, sp)


def f(P, mode='optimization'):

  S,R,target = np.zeros(T),np.zeros(T),np.zeros(T)
  cost = 0
  S[0] = df.storage.values[0]
  policies = [None]

  for t in range(1,T):

    TDI = np.sum(Q[t+1:t+4])
    policy = P.evaluate([S[t-1], dowy[t], TDI])

    if policy == 'Release_Demand':
      target[t] = D[t]
    elif policy == 'Hedge_80':
      target[t] = 0.8*D[t]
    elif policy == 'Hedge_50':
      target[t] = 0.5*D[t]
    elif policy == 'Flood_Control':
      if fit_historical:
        target[t] = max(0.2*(Q[t] + S[t-1] - tocs(dowy[t])), 0.5*D[t])
      else:
        target[t] = max(S[t-1] + TDI - K, 0)

    if mode == 'simulation':
      policies.append(policy)

    # max/min release
    R[t] = min(target[t], S[t-1] + Q[t])
    R[t] = min(R[t], max_release(S[t-1]))
    R[t] +=  max(S[t-1] + Q[t] - R[t] - K, 0) # spill
    S[t] = S[t-1] + Q[t] - R[t]

    # squared deficit. Also penalize any total release over 100 TAF/day  
    # should be able to vectorize this.  
    cost += max(D[t] - R[t], 0)**2 + max(R[t]-100, 0)**2


  if mode == 'simulation':
    df['Ss'] = pd.Series(S, index=df.index)
    df['Rs'] = pd.Series(R, index=df.index)
    df['demand'] = pd.Series(D, index=df.index)
    df['target'] = pd.Series(target, index=df.index)
    df['policy'] = pd.Series(policies, index=df.index, dtype='category')
    return df
  else:
    if fit_historical:
      return np.sqrt(np.mean((S - df.storage.values)**2))
    else:
      return cost


def init_plotting():
  sns.set_style('whitegrid')

  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (12, 8)
  plt.rcParams['font.size'] = 13
  plt.rcParams['font.family'] = 'OfficinaSanITCBoo'
  # plt.rcParams['font.weight'] = 'bold'
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


def plot_results(df, filename = None, dpi = 300):

  fig = plt.figure(figsize=(12, 6)) 
  gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1]) 

  ax0 = plt.subplot(gs[0])
  df.storage.plot(ax=ax0, color='k')
  df.Ss.plot(ax=ax0, color='indianred')
  ax0.set_ylabel('')
  ax0.xaxis.grid(False)
  ax0.set_title('Storage, TAF', family='OfficinaSanITCMedium', loc='left')
  ax0.legend(['Observed', 'Simulated'], loc=3, ncol=3)
  ax0.set_ylim([0,1000])

  ax1 = plt.subplot(gs[1])
  ax1.scatter(df.storage.values, df.Ss.values, s=3, c='steelblue', edgecolor='none', alpha=0.7)
  ax1.set_xlim([0,1000])
  ax1.set_ylim([0,1000])
  ax1.set_ylabel('Simulated (TAF)')
  ax1.set_xlabel('Observed (TAF)')
  ax1.plot([0,1000],[0,1000], color='k')
  ax1.annotate('$R^2 = %0.2f$' % np.corrcoef(df.Ss.values, df.storage.values)[0,1], xy=(600,100), color='0.3')

  ax2 = plt.subplot(gs[2])
  np.log10(taf_to_cfs(df.outflow)).plot(color='k', ax=ax2)
  np.log10(taf_to_cfs(df.Rs)).plot(color='indianred', ax=ax2)
  # taf_to_cfs(df.inflow).plot(color='blue')
  # df.demand.plot(color='green', linewidth=2)
  ax2.set_ylabel('')
  ax2.xaxis.grid(False)
  ax2.set_title('Release, log$_{10}$ cfs', family='OfficinaSanITCMedium', loc='left')
  ax2.legend(['Observed', 'Simulated'], loc=3, ncol=3)
  ax2.set_ylim([1.5,5.5])

  ax3 = plt.subplot(gs[3])
  r2 = np.corrcoef(df.Rs.values, df.outflow.values)[0,1]
  ax3.scatter(np.log10(taf_to_cfs(df.outflow)), np.log10(taf_to_cfs(df.Rs)), s=3, c='steelblue', edgecolor='none', alpha=0.7)
  ax3.set_xlim([2.5,5.5])
  ax3.set_ylim([2.5,5.5])
  ax3.plot([1.0,5.5],[1.0,5.5], color='k')
  ax3.set_ylabel('Simulated (log cfs)')
  ax3.set_xlabel('Observed (log cfs)')
  ax3.annotate('$R^2 = %0.2f$' % r2, xy=(4.25,2.75), color='0.3')

  plt.tight_layout()

  if filename is None:
    plt.show()
  else:
    plt.savefig(filename, dpi=dpi)
    plt.close()


# run this stuff whenever the file is imported
df = pd.read_csv('folsom-daily.csv', index_col=0, parse_dates=True)
df = df['1995-10-01':'2015-09-30']
Q = df.inflow.values
K = 975 # capacity, TAF
dowy = np.array([water_day(d) for d in df.index.dayofyear])
D = 4 + 3*np.sin(2*np.pi*dowy/365 - np.pi)
T = len(Q)

fit_historical = True
init_plotting()

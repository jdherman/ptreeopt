import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
# from folsom import Folsom


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


init_plotting(5, 6)

# Found using a different script. These scenarios have no flood control costs.
adaptable_scenarios = ['access1-0_rcp45_r1i1p1', 'access1-0_rcp85_r1i1p1',
                       'bcc-csm1-1-m_rcp45_r1i1p1', 'bcc-csm1-1-m_rcp85_r1i1p1', 'bcc-csm1-1_rcp26_r1i1p1',
                       'bcc-csm1-1_rcp45_r1i1p1', 'bcc-csm1-1_rcp60_r1i1p1', 'ccsm4_rcp26_r1i1p1', 'ccsm4_rcp45_r1i1p1',
                       'cesm1-cam5_rcp26_r1i1p1', 'cesm1-cam5_rcp60_r1i1p1', 'cesm1-cam5_rcp85_r1i1p1', 'cmcc-cm_rcp45_r1i1p1',
                       'cmcc-cm_rcp85_r1i1p1', 'csiro-mk3-6-0_rcp26_r1i1p1', 'csiro-mk3-6-0_rcp45_r1i1p1', 'csiro-mk3-6-0_rcp60_r1i1p1',
                       'csiro-mk3-6-0_rcp85_r1i1p1', 'fgoals-g2_rcp26_r1i1p1', 'fgoals-g2_rcp45_r1i1p1', 'fio-esm_rcp26_r1i1p1',
                       'fio-esm_rcp60_r1i1p1', 'fio-esm_rcp85_r1i1p1', 'gfdl-cm3_rcp26_r1i1p1', 'gfdl-cm3_rcp45_r1i1p1',
                       'gfdl-cm3_rcp60_r1i1p1', 'gfdl-cm3_rcp85_r1i1p1', 'gfdl-esm2g_rcp45_r1i1p1', 'gfdl-esm2m_rcp26_r1i1p1',
                       'gfdl-esm2m_rcp45_r1i1p1', 'gfdl-esm2m_rcp60_r1i1p1', 'giss-e2-h-cc_rcp45_r1i1p1', 'giss-e2-r-cc_rcp45_r1i1p1',
                       'giss-e2-r_rcp26_r1i1p1', 'giss-e2-r_rcp45_r1i1p1', 'giss-e2-r_rcp60_r1i1p1', 'giss-e2-r_rcp85_r1i1p1',
                       'hadgem2-ao_rcp26_r1i1p1', 'hadgem2-ao_rcp60_r1i1p1', 'hadgem2-cc_rcp85_r1i1p1', 'hadgem2-es_rcp26_r1i1p1',
                       'hadgem2-es_rcp45_r1i1p1', 'hadgem2-es_rcp60_r1i1p1', 'inmcm4_rcp45_r1i1p1', 'ipsl-cm5a-mr_rcp26_r1i1p1',
                       'ipsl-cm5b-lr_rcp85_r1i1p1', 'miroc5_rcp26_r1i1p1', 'miroc5_rcp45_r1i1p1', 'miroc5_rcp60_r1i1p1',
                       'miroc5_rcp85_r1i1p1', 'miroc-esm-chem_rcp26_r1i1p1', 'miroc-esm-chem_rcp45_r1i1p1', 'miroc-esm-chem_rcp60_r1i1p1',
                       'miroc-esm_rcp26_r1i1p1', 'miroc-esm_rcp45_r1i1p1', 'miroc-esm_rcp85_r1i1p1', 'mpi-esm-lr_rcp26_r1i1p1',
                       'mpi-esm-mr_rcp26_r1i1p1', 'mpi-esm-mr_rcp45_r1i1p1', 'mpi-esm-mr_rcp85_r1i1p1', 'mri-cgcm3_rcp26_r1i1p1',
                       'mri-cgcm3_rcp45_r1i1p1', 'mri-cgcm3_rcp85_r1i1p1', 'noresm1-m_rcp26_r1i1p1', 'noresm1-m_rcp60_r1i1p1',
                       'noresm1-m_rcp85_r1i1p1']

##############################
# STEP 1: All of the validation runs (optimized in s1, evaluated in s2) plus historical
# first get cc-opt results
# then get the re-evaluated historical policy on the CC scenarios
# also re-evaluate the robust policy in every scenario. Save a pickle.

# d = []

# # scenarios = pd.read_csv('folsom/data/folsom-cc-inflows.csv', index_col=0).columns
annQs = pd.read_csv('../folsom/data/folsom-cc-annQ-MA30.csv',
                    index_col=0, parse_dates=True)
# hist_snapshots = pickle.load(open('results/hist-tocs/snapshots-tocs-depth-3-seed-0.pkl', 'rb'))
# histP = hist_snapshots['best_P'][-1]
# cc_snapshots = pickle.load(open('results/cc-full/snapshots-cc-full-4.pkl', 'rb'))
# robustP = cc_snapshots['best_P'][-1]

# for s in adaptable_scenarios:
#   print s
#   df = pickle.load(open('results/cc-opt/snapshots-cc-' + s + '.pkl', 'rb'))
#   f_cc_opt = df['best_f'][-1]

#   tocs_model = Folsom('folsom/data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
#               scenario = s, cc = True, use_tocs=True)
#   f_baseline = tocs_model.f(histP)

#   model = Folsom('folsom/data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
#               scenario = s, cc = True, use_tocs=False)
#   f_cc_full = model.f(robustP)

#   d.append({'scenario': s, 'type': 'Optimized', 'J': f_cc_opt})
#   d.append({'scenario': s, 'type': 'Baseline', 'J': f_baseline})
#   d.append({'scenario': s, 'type': 'Adaptive', 'J': f_cc_full})

# pickle.dump(d, open('cc-hist-val.pkl', 'w'))


# STEP 3: Plot the thing
# **But only for adapt-able scenarios (no flood cost)


d = pickle.load(open('validations/cc-hist-val.pkl', 'rb'))
df = pd.DataFrame(d)
df = df[df.scenario.isin(adaptable_scenarios)]
df.J[df.J > 4] = 4.0

annQs = annQs[df.scenario.unique()]['2099'].max(axis=0).sort_values()
sns.stripplot(data=df, x='J', y='scenario', hue='type',
              order=annQs.index.values, split=False, edgecolor='none',
              palette=['gold', '0.25', 'royalblue'])

# plt.xscale('log')

for i, s in enumerate(annQs.index.values):
    points = df[df.scenario == s].J.values
    plt.plot([min(points), max(points)], [i, i], color='0.75', linewidth=0.5)

plt.xlim([-0.1, 4.1])
# plt.ylim([-1,67])
plt.gca().set_yticklabels([])
plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(False)
sns.despine(left=False, bottom=False)
# plt.axvline(10.0, color='k', linestyle='--', linewidth=1)
plt.ylabel('Scenario (Wet --> Dry)')
plt.xlabel('J (TAF/d)$^2$')
plt.tight_layout()
# plt.show()
plt.savefig('stripplot.svg')

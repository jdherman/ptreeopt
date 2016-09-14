import numpy as np
# import matplotlib.pyplot as plt
import pickle
from opt import *
from folsom import Folsom
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD

np.random.seed(comm.rank)

# these are the scenarios that can self-optimize
# found with print-cc-results.py

scenarios = ['access1-0_rcp45_r1i1p1', 'access1-0_rcp85_r1i1p1', 
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

model = Folsom('data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
               cc = True, use_tocs=False)

def wrapper(P):

  J = 0
  for s in scenarios:
    model.set_scenario(s)
    J += model.f(P)

  return (J / len(scenarios))



algorithm = PTreeOpt(wrapper,    # removed "Inflow" from [0,300], and "WYC" from [120,190]
                    feature_bounds = [[0,1000], [1,365], [0,300],  [1500,4000]],# [100,350]],
                    feature_names = ['Storage', 'Day', 'Inflow', 'AnnQ'],#, 'LP3'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 5
                    )


snapshots = algorithm.run(max_nfe = 6000, log_frequency = 50)
pickle.dump(snapshots, open('output/snapshots-cc-full-' + s + '.pkl', 'wb'))

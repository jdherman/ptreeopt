import numpy as np
# import matplotlib.pyplot as plt
import pickle
from folsom import Folsom
from ptreeopt import PTreeOpt
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

model = Folsom('folsom/data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
               cc = True, use_tocs=False)

def wrapper(P):

  J = 0
  for s in scenarios:
    model.set_scenario(s)
    J += model.f(P)

  return (J / len(scenarios))



algorithm = PTreeOpt(wrapper,    # added all back in 9/26/16
                    feature_bounds = [[0,1000], [0,100], [0,365], [2250,3250], [150,250], [140,165]],
                    feature_names = ['Storage', 'Inflow', 'Day', 'AnnQ', 'LP3', 'WYC'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 20,
                    cx_prob = 0.70,
                    population_size = 100,
                    max_depth = 5
                    )


snapshots = algorithm.run(max_nfe = 3000, log_frequency = 50)
pickle.dump(snapshots, open('output/snapshots-cc-full-' + str(comm.rank) + '.pkl', 'wb'))

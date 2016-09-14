import numpy as np
# import matplotlib.pyplot as plt
import pickle
from opt import *
from folsom import Folsom
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD

np.random.seed(1)

scenarios = pd.read_csv('data/folsom-cc-inflows.csv', index_col=0).columns
s = scenarios[comm.rank]

# set back to 1999 if using all of the input variables
model = Folsom('data/folsom-cc-inflows.csv', sd='2050-10-01', ed='2099-09-30',
                scenario = s, cc = True)

algorithm = PTreeOpt(model.f,    # removed "Inflow" from [0,300], and "WYC" from [120,190]
                    feature_bounds = [[0,1000], [1,365], [0,300]],#  [1500,4000], [100,350]],
                    feature_names = ['Storage', 'Day', 'Inflow'],# 'AnnQ', 'LP3'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 5
                    )

snapshots = algorithm.run(max_nfe = 25000, log_frequency = 50)
pickle.dump(snapshots, open('output/snapshots-cc-' + s + '.pkl', 'wb'))

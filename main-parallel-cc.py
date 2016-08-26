import numpy as np
# import matplotlib.pyplot as plt
import pickle
from opt import *
import folsom_cc
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD

np.random.seed(comm.rank)

algorithm = PTreeOpt(folsom_cc.f, 
                    feature_bounds = [[0,1000], [1,365], [1500,4000], [100,350], [120,190]],
                    feature_names = ['Storage', 'Day', 'AnnQ', 'LP3', 'WYC'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 4
                    )

snapshots = algorithm.run(max_nfe = 10000, log_frequency = 50)
pickle.dump(snapshots, open('output/snapshots-cc-seed-' + str(comm.rank) + '.pkl', 'wb'))

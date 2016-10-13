import numpy as np
import matplotlib.pyplot as plt
import pickle
from folsom import Folsom
from ptreeopt import PTreeOpt
import pandas as pd

np.random.seed(13)

model = Folsom('folsom/data/folsom-daily.csv', sd='1995-10-01', ed='2015-09-30', use_tocs = False)

algorithm = PTreeOpt(model.f, 
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'Inflow'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 20,
                    cx_prob = 0.70,
                    population_size = 100,
                    max_depth = 7
                    )


snapshots = algorithm.run(max_nfe = 50000, log_frequency = 100)
# pickle.dump(snapshots, open('snapshots-hist-opt.pkl', 'wb'))

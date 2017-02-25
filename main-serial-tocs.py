import numpy as np
import matplotlib.pyplot as plt
import pickle
from ptreeopt.opt import *
from folsom import Folsom
import pandas as pd

np.random.seed(8)

model = Folsom('folsom/data/folsom-daily-w2016.csv', sd='1995-10-01', ed='2015-09-30', use_tocs = True)

algorithm = PTreeOpt(model.f, 
                    feature_bounds = [[0,1000], [1,365]],# [0,300]],
                    feature_names = ['Storage', 'Day'],# 'Inflow'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 5
                    )


snapshots = algorithm.run(max_nfe = 1000, log_frequency = 50)
pickle.dump(snapshots, open('snapshots-hist-tocs-test.pkl', 'wb'))

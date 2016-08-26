import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from opt import *
import folsom


algorithm = PTreeOpt(folsom.f, 
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'TDI'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_80', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 4
                    )

snapshots = pickle.load(open('results/historical-training/snapshots-depth-3-seed-0.pkl', 'rb'))

P1 = snapshots['best_P'][-1]
# print P1

snapshots = pickle.load(open('results/historical-training/snapshots-depth-3-seed-18.pkl', 'rb'))

# P2 = snapshots['best_P'][-1]
# print P2

# P3,P4 = algorithm.crossover(P1,P2)
# P3.prune()
# P4.prune()
# print P3
# print P4

# CROSSOVER PLOTS
# P1.graphviz_export('figs/fig-3-operators/crossover-P1.svg')
# P2.graphviz_export('figs/fig-3-operators/crossover-P2.svg')
# P3.graphviz_export('figs/fig-3-operators/crossover-P3.svg')
# P4.graphviz_export('figs/fig-3-operators/crossover-P4.svg')

# MUTATION PLOTS
# P5 = algorithm.mutate(P1)
# P5.prune()
# print P5

# P1.graphviz_export('figs/fig-3-operators/mutation-P1.svg')
# P5.graphviz_export('figs/fig-3-operators/mutation-P5.svg')


# PRUNING PLOTS
T = algorithm.random_tree() # baked in
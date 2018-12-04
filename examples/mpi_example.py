import sys
sys.path.append('..')

import numpy as np
from .folsom import Folsom
from ptreeopt import PTreeOpt, MPIExecutor

# Example to run optimization and save results
np.random.seed(17)

model = Folsom('folsom/data/folsom-daily-w2016.csv',
               sd='1995-10-01', ed='2016-09-30', use_tocs=False)

def run(P):
    # model.f has side effects: it changes values on P
    # so for parallel running with multiprocessing, we want to return
    # also the modified P
    results = model.f(P)
    return P, results

algorithm = PTreeOpt(model.f,
                     feature_bounds=[[0, 1000], [1, 365], [0, 300]],
                     feature_names=['Storage', 'Day', 'Inflow'],
                     discrete_actions=True,
                     action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
                                   'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                     mu=20,
                     cx_prob=0.70,
                     population_size=100,
                     max_depth=5
                     )

if __name__ == '__main__':
        
    # With only 1000 function evaluations this will not be very good
    # should be run with mpiexec -n 6 python -m mipy4py.futures mpi_example.py
    with MPIExecutor(algorithm) as executor:
        snapshots = executor.run(1000, 10)

# pickle.dump(snapshots, open('example-results.pkl', 'wb'))
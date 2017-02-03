import numpy as np
import pickle
from folsom import Folsom

seeds = 50
Jv = np.zeros((50,10))

for depth in range(1,10):

  for s in range(seeds):
    model = Folsom('folsom-daily.csv', sd='1955-10-01', ed='1995-09-30', fit_historical = False)
    fname = 'snapshots-depth-' + str(depth) + '-seed-' + str(s) + '.pkl'
    data = pickle.load(open('results/hist-opt/' + fname, 'rb'))
    P = data['best_P'][-1]
    Jv[s,depth] = model.f(P)
    print Jv[s,depth]
    # print 'Historical: ' + str(data['best_f'][-1]) + ', Validation: ' + str(Jv[s])

np.savetxt('results/historical-validation/results.csv', Jv, delimiter=',')


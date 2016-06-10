import numpy as np 
import matplotlib.pyplot as plt

from opt import *

# algorithm = PTreeOpt()

def f(P, plot=False):
  T = 100
  Q = np.random.normal(50,10,T)
  R = np.zeros_like(Q)
  S = np.zeros_like(Q)
  K = 900
  S[0] = 500
  obj = 0

  for t in range(T):
    W = S[t-1] + Q[t]
    R[t] = min(P.evaluate([W]), W)
    S[t] = min(W - R[t], K)
    obj += R

  if plot:
    plt.plot(S+Q)
    plt.plot(R)
    plt.show()

  return -obj

# 1 feature: res. storage
feature_bounds = [(0,1000)]
action_bounds = [(0,100)]

algorithm = PTreeOpt(f, 1000, feature_bounds, action_bounds)

algorithm.initialize()
# print algorithm.objectives

algorithm.population[99].graphviz_export('graphviz/test')

f(algorithm.population[99], plot=True)




# Need to add:
# feature variable names
# action variable name
# discrete vs. continuous actions (it matters)
# operators (!!) mutation first

# to validate a tree ... this will be important
# lower nodes/logic cannot contradict parent logic
# or else some branches will always evaluate false
# lots of other "validity" gotchas.  


# ideas from DEAP: 
# depth-first list representation of a tree (allows for consecutive sub-trees)
# one-point crossover mutation/slicing


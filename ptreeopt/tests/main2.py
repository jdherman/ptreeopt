import numpy as np
import matplotlib.pyplot as plt

from opt import *

# algorithm = PTreeOpt()


def f(P, plot=False):
    T = 100
    Q = np.random.normal(50, 10, T)
    R = np.zeros_like(Q)
    S = np.zeros_like(Q)
    K = 900
    S[0] = 500
    obj = 0

    for t in range(T):
        W = S[t - 1] + Q[t]
        R[t] = min(P.evaluate([W]), W)
        S[t] = min(W - R[t], K)
        obj += R[t]

    if plot:
        plt.plot(S + Q)
        plt.plot(R)
        plt.show()

    return -obj  # minimizes by default


# 1 feature: res. storage
feature_bounds = [[0, 1000]]
names = ['Storage']
action_bounds = [0, 100]

algorithm = PTreeOpt(f, 10000, feature_bounds, action_bounds,
                     feature_names=names, population_size=100)

algorithm.initialize()
# print algorithm.objectives

algorithm.run()
print algorithm.best_P
print algorithm.best_f
# print [i for i in algorithm.population[0].L]
# f(algorithm.population[0], plot=True)
f(algorithm.best_P, plot=True)

algorithm.best_P.graphviz_export('graphviz/bestP')


# Need to add:


# feature variable names
# action variable name
# to make these work, they must be properties of the nodes

# discrete vs. continuous actions (it matters) -- not right now.
# operators (!!) mutation first

# to validate a tree ... this will be important
# lower nodes/logic (should not) contradict parent logic
# or else some branches will always evaluate false (this might be ok)

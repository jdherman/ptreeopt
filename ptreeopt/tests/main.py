import numpy as np
import matplotlib.pyplot as plt

from tree import *
from opt import *


# algorithm = PTreeOpt()
L = [[0, 5], [3], [1, 15], [1], [2.50]]
# L = [[0, 27.913518267933203], [0, 76.70516906882096], [6.608787642015461], [8.389976574757554], [6.738244179343229]]

# L = [(1,10), (5,12), (3,15), (2,25), (4,30), (0,36)]
# L = [(1,10), (None,25), (None,36)]
T = PTree(L)  # do this once at the beginning of the simulation

# print T.L
# L = gen_random_tree_as_list(3, 5)
print T.evaluate(states=[5, 17, 100])  # do this every timestep
T.graphviz_export('graphviz/smalltest')

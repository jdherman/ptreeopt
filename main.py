import numpy as np 
import matplotlib.pyplot as plt

from tree import *

L = [(0,5), (None,3), (1,15), None, None, (None,1), (None,2.50)]
# L = [(1,10), (5,12), (3,15), (2,25), (4,30), (0,36)]
# L = [(1,10), (None,25), (None,36)]
T = build_tree(L, export_png=True) # do this once at the beginning of the simulation
# L = gen_random_tree_as_list(3, 5)
print eval_tree(T, states=[6,16,100]) # do this every timestep


# Need to add:
# feature variable names
# action variable name
# discrete vs. continuous actions (it matters)
# operators
# feature/action bounds scaling

# to validate a tree ... this will be important
# lower nodes/logic cannot contradict parent logic
# or else it will always evaluate false
# lots of other "validity" gotchas. 
import numpy as np 
import matplotlib.pyplot as plt

TEST ONE
L = [['Flood_Control']]
L = [[1,256], ['Flood_Control'], ['Release_Demand']]
P = PTree(L)
# P.graphviz_export('graphviz/whatever.png')
results = folsom.f(P, mode='simulation')
folsom.plot_results(results)
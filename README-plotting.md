This readme describes plotting functions after the optimization is done.

```python
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from ptreeopt.plotting import *
from folsom import Folsom
```

First load the results. `snapshots` is a dictionary containing three lists: `best_P` (the best policies throughout the optimization), `best_f` (the best objective function values throughout the optimization), and `nfe` (the number of function evaluations).

```python
snapshots = pickle.load(open('test-results.pkl', 'rb'), encoding='latin1')
P = snapshots['best_P'][0]
print(P)
```

Two ways to look at trees: either one static plot, or an animation. To add colors to the action nodes, create a dictionary like this where the keys are the actions. Note the animation requires ImageMagick, and the static tree requires pygraphviz.

```python

colors = {'Release_Demand': 'cornsilk',
          'Hedge_90': 'cornsilk',
          'Hedge_80': 'indianred',
          'Hedge_70': 'indianred',
          'Hedge_60': 'indianred',
          'Hedge_50': 'indianred',
          'Flood_Control': 'lightsteelblue'}

graphviz_export(P, 'test.svg', colordict=colors) # creates one SVG

animate_trees(snapshots, 'tree', colordict=colors) # creates tree.gif
```

To plot the timeseries of actions taken by the policy during the model simulation, we have to rerun the simulation and save the results. This is somewhat model-specific, but the `ts_color` function will create a plot with colors in the background based on the pandas series `df.policy`. It does not save to a file because we probably want to plot something else on it too.

```python
model = Folsom('folsom/data/folsom-daily-w2016.csv',
                    sd='1995-10-01', ed='2015-09-30', use_tocs=False)
df = model.f(P, mode='simulation')
ts_color(df.policy, colordict=colors)
plt.show()
```

Creating only a single static plot of the objective function value is easy, just plot `snapshots['best_f']` against `snapshots['nfe']`. But to go along with the tree animation, we might also want to animate the objective function value during the search:

```python
animate_objfxn(snapshots, 'objfxn') # creates objfxn.gif
```


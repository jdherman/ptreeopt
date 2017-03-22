## Policy Tree Optimization

Heuristic policy search for control of dynamic systems. Uses genetic programming to develop binary trees relating observed indicator variables to actions, either real-valued or discrete. A simulation model serves as the objective function. 

**Requirements:** [NumPy](http://www.numpy.org/), [PyGraphviz](https://pygraphviz.github.io/) (optional). The example model also uses [pandas](http://pandas.pydata.org/) and [Matplotlib](http://matplotlib.org/) but these are not strictly required.

Still in active development, not stable. So far only tested with Python 2.7. Contributions and bug reports welcome!

**Citation:** Paper in review. The full set of experiments and data are available in the [paper branch](https://github.com/jdherman/ptreeopt/tree/paper).

### Example
This example develops a control policy based on a simulation model of Folsom Reservoir.
```python
import numpy as np
from folsom import Folsom
from ptreeopt import PTreeOpt

np.random.seed(17) # to be able to replicate the results

model = Folsom('folsom/data/folsom-daily-w2016.csv', 
                sd='1995-10-01', ed='2016-09-30', use_tocs = False)

algorithm = PTreeOpt(model.f, 
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'Inflow'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 
                    'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 20,
                    cx_prob = 0.70,
                    population_size = 100,
                    max_depth = 7
                    )

snapshots = algorithm.run(max_nfe = 1000, log_frequency = 100)
```

`model.f` is a simulation model that will be evaluated many times. The simulation model must be set up like this:
```python
def f(P):
  # ...
  for t in range(T):
    # observe indicators x1,x2,x3
    action = P.evaluate([x1,x2,x3]) # returns a string from `action_names`

    if action == 'something':
      # ...
    if action == 'something_else':
      # ...
  
  return objective # assumes minimization
```

The Folsom simulation model ([link](https://github.com/jdherman/ptreeopt/blob/master/folsom/folsom.py)) gives a more detailed example. 

`snapshots` is a dictionary containing keys `'nfe', 'time', 'best_f', 'best_P'` (number of function evaluations, elapsed time, best objective function value, and best policy tree). Each key points to a list of length `max_nfe/log_frequency`. 

To save an image of the best policy (requires PyGraphviz):
```python
P = snapshots['best_P'][-1]
P.graphviz_export('filename.svg')
```

or save all results as a pickle for later analysis:
```python
import pickle
pickle.dump(snapshots, open('snapshots.pkl', 'wb'))
```

### License
Copyright (C) 2017 Jon Herman and Matteo Giuliani. Released under the [MIT license](LICENSE.md).

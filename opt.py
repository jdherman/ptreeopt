from __future__ import division
import numpy as np
import time, datetime, copy
from tree import *


class PTreeOpt():

  def __init__(self, f, feature_bounds, discrete_actions = False, action_bounds = None, 
               action_names = None, population_size = 100, mu = 15, 
               max_depth = 4, mut_prob = 0.9, cx_prob = 0.9, feature_names = None):

    self.f = f
    self.num_features = len(feature_bounds)
    self.feature_bounds = feature_bounds
    self.discrete_actions = discrete_actions
    self.action_bounds = action_bounds
    self.action_names = action_names
    self.popsize = population_size
    self.mu = mu
    self.max_depth = max_depth
    self.mut_prob = mut_prob
    self.cx_prob = cx_prob
    self.feature_names = feature_names

    if feature_names is not None and len(feature_names) != len(feature_bounds):
      raise ValueError('feature_names and feature_bounds must be the same length.')

    if discrete_actions:
      if action_names is None or action_bounds is not None:
        raise ValueError('''discrete_actions must be run with action_names, 
        (which are strings), and not action_bounds.''')
    else:
      if action_bounds is None:
        raise ValueError('''Real-valued actions (which is the case by 
        default, discrete_actions=False) must include action_bounds. 
        Currently only one action is supported, so bounds = [lower, upper].''')

    if mu > population_size:
      raise ValueError('''Number of parents (mu) cannot be greater than 
      the population_size.''')


  def iterate(self):

    ix = np.argsort(self.objectives)
    self.population = self.population[ix] 
    self.objectives = self.objectives[ix]

    if self.best_f is None or self.objectives[0] < self.best_f:
      self.best_f = self.objectives[0]
      self.best_P = self.population[0]

    for i in range(self.mu, self.popsize):
      
      if np.random.rand() < self.cx_prob:
        P1,P2 = np.random.choice(self.population[:self.mu], 2, replace=False)
        child = self.crossover(P1,P2)[0]

        # bloat control
        while child.get_depth() > self.max_depth:
          child = self.crossover(P1,P2)[0]

      else: # replace with random new tree
        child = self.random_tree()

      self.population[i] = self.mutate(child)
      self.objectives[i] = self.f(self.population[i])


  def run(self, max_nfe = 100, log_frequency = None):
            
    start_time = time.time()
    nfe,last_log = 0,0

    self.population = np.array([self.random_tree() for _ in range(self.popsize)])
    self.objectives = np.array([self.f(P) for P in self.population])
    self.best_f = None
    self.best_P = None
    
    if log_frequency:
      print 'NFE\telapsed_time\tbest_f'

    while nfe < max_nfe:
      self.iterate()
      nfe += self.popsize

      if log_frequency is not None and nfe >= last_log + log_frequency:
        elapsed = datetime.timedelta(seconds=time.time()-start_time).seconds
        print '%d\t%s\t%0.3f\t%s' % (nfe, elapsed, self.best_f, self.best_P)        
        last_log = nfe    
  

  def random_tree(self, terminal_ratio = 0.5):
    depth = np.random.randint(2, self.max_depth+1)
    L = []
    S = [0]

    while S:
      current_depth = S.pop()

      # action node
      if current_depth == depth or (current_depth > 0 and np.random.rand() < terminal_ratio):
        if self.discrete_actions:
          L.append([str(np.random.choice(self.action_names))])
        else:
          L.append([np.random.uniform(*self.action_bounds)])

      else:
        x = np.random.choice(self.num_features)
        v = np.random.uniform(*self.feature_bounds[x])
        L.append([x,v])
        S += [current_depth+1]*2

    return PTree(L, self.feature_names)


  def crossover(self, P1, P2):
    P1,P2 = [copy.deepcopy(P) for P in P1,P2]
    # should use indices of ONLY feature nodes
    feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
    feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
    index1 = np.random.choice(feature_ix1)
    index2 = np.random.choice(feature_ix2)
    slice1 = P1.get_subtree(index1)
    slice2 = P2.get_subtree(index2)
    P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
    P1.build()
    P2.build()
    P1.prune()
    P2.prune()
    return (P1,P2)


  def mutate(self, P):
    P = copy.deepcopy(P)

    for item in P.L:
      if np.random.rand() < self.mut_prob:
        if item.is_feature:
          item.threshold = self.bounded_gaussian(item.threshold, self.feature_bounds[item.index])
        else:
          if self.discrete_actions:
            item.value = str(np.random.choice(self.action_names))
          else:
            item.value = self.bounded_gaussian(item.value, self.action_bounds)

    return P


  def bounded_gaussian(self, x, bounds):
    # do mutation in normalized [0,1] to avoid sigma scaling issues
    lb,ub = bounds
    xnorm = (x-lb)/(ub-lb)
    x_trial = xnorm + np.random.normal(0, scale=0.1)
    
    while np.any((x_trial > 1) | (x_trial < 0)):
      x_trial = xnorm + np.random.normal(0, scale=0.1)
    
    return lb + x_trial*(ub-lb)



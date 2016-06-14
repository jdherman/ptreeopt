from __future__ import division
import numpy as np
import time, datetime, copy
from tree import *

# want to use Dave's "solution" and "problem" classes?

class PTreeOpt():

  def __init__(self, f, max_NFE, feature_bounds, action_bounds, 
               population_size = 100, mu = 15, max_depth = 4, mut_prob = 0.9, 
               feature_names = None, log_frequency = None):

    self.f = f
    self.max_NFE = max_NFE
    self.num_features = len(feature_bounds)
    self.feature_bounds = feature_bounds
    self.action_bounds = action_bounds
    self.popsize = population_size
    self.mu = mu
    self.max_depth = max_depth
    self.mut_prob = 0.9
    self.feature_names = feature_names
    self.nfe = 0
    self.log_frequency = log_frequency


  def initialize(self):
    self.population = [self.random_tree() for _ in range(self.popsize)]
    self.objectives = [self.f(P) for P in self.population]
    self.best_f = None
    self.best_P = None


  def iterate(self):

    ix = np.argsort(self.objectives)[:self.mu]

    if self.best_f is None or self.objectives[ix[0]] < self.best_f:
      self.best_f = self.objectives[ix[0]]
      self.best_P = self.population[ix[0]]
    parents = [self.population[i] for i in ix]
    
    for i in range(self.popsize):
      pair = np.random.choice(parents, 2, replace=False)
      child = self.crossover(*pair)[0]

      # bloat control
      if child.get_depth() > self.max_depth:
        child = np.random.choice(pair)

      self.population[i] = self.mutate(child)

    self.objectives = [self.f(P) for P in self.population]
    self.nfe += self.popsize


  def run(self):
            
    start_time = time.time()
    last_log = self.nfe
    
    self.initialize()

    if self.log_frequency:
      print 'NFE\tTime (s)\tBest f'

    while not self.nfe >= self.max_NFE:
      self.iterate()

      if self.log_frequency is not None and self.nfe >= last_log + self.log_frequency:
        elapsed = datetime.timedelta(seconds=time.time()-start_time).seconds
        print '%d\t%s\t%0.3f\t%s' % (self.nfe, elapsed, self.best_f, self.best_P)        
        last_log = self.nfe    
          

    # save one last point here in stats
    # self.nfe,
    # datetime.timedelta(seconds=time.time()-start_time))
  

  def random_tree(self, ratio = 0.5):
    depth = np.random.randint(2, self.max_depth+1)
    L = []
    S = [0]

    while S:
      current_depth = S.pop()

      # action node
      if current_depth == depth or (current_depth > 0 and np.random.rand() < ratio):
        L.append([np.random.uniform(*self.action_bounds)])

      else:
        x = np.random.choice(self.num_features)
        v = np.random.uniform(*self.feature_bounds[x])
        L.append([x,v])
        S += [current_depth+1]*2

    return PTree(L, self.feature_names)


  def crossover(self, P1, P2):
    P1 = copy.deepcopy(P1)
    P2 = copy.deepcopy(P2)
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
    return (P1,P2)


  def mutate(self, P):
    P = copy.deepcopy(P)

    for item in P.L:
      if np.random.rand() < self.mut_prob:
        if item.is_feature:
          item.threshold = self.bounded_gaussian(item.threshold, self.feature_bounds[item.index])
        else:
          item.value = self.bounded_gaussian(item.value, self.action_bounds)

    return P


  def bounded_gaussian(self, x, bounds):
    # do mutation in normalized [0,1] to avoid specifying sigma
    lb,ub = bounds
    xnorm = (x-lb)/(ub-lb)
    x_trial = xnorm + np.random.normal(0, scale=0.1)
    
    while np.any((x_trial > 1) | (x_trial < 0)):
      x_trial = xnorm + np.random.normal(0, scale=0.1)
    
    return lb + x_trial*(ub-lb)



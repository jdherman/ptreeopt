from __future__ import division
import numpy as np
import time, datetime, copy
from .tree import *


class PTreeOpt():

  def __init__(self, f, feature_bounds, discrete_actions = False, action_bounds = None, 
               action_names = None, population_size = 100, mu = 15, max_depth = 4, 
               mut_prob = 0.9, cx_prob = 0.9, feature_names = None, 
               multiobj = False, epsilons = None):

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
    self.multiobj = multiobj
    self.epsilons = epsilons

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

    # selection: find index numbers for parents
    if not self.multiobj:
      parents = self.select_truncation(self.objectives)

      if self.best_f is None or self.objectives[parents[0]] < self.best_f:
        self.best_f = self.objectives[parents[0]]
        self.best_P = self.population[parents[0]]
    else:
      parents = [self.binary_tournament(self.population, self.objectives)
                 for _ in range(self.mu)]

      if self.best_f is None:
        self.best_f = self.objectives[parents]
        self.best_P = self.population[parents]
      else:
        self.best_P,self.best_f = self.archive_sort(self.best_P, self.best_f, 
                                                    self.population, self.objectives)

    # first: mutate the parents, only keep child if it's better
    # changed 7/17/17 JDH: now mutate all except best parent (single-obj)
    children = set(range(self.popsize)) - set(parents)

    for i in parents[1:]:
      child = self.mutate(self.population[i])
      child.prune()
      self.population[i] = child

    # then crossover to develop the rest of the population
    for i in children:
      
      if np.random.rand() < self.cx_prob:
        P1,P2 = np.random.choice(self.population[parents], 2, replace=False)
        child = self.crossover(P1,P2)[0]

        # bloat control
        while child.get_depth() > self.max_depth:
          child = self.crossover(P1,P2)[0]

      else: # replace with randomly chosen parent
        child = np.random.choice(self.population[parents], 1)[0]

      child = self.mutate(child)
      child.prune()
      self.population[i] = child


  def run(self, max_nfe=100, parallel=False, log_frequency=None):
    
    if parallel:
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      size = comm.Get_size()
      rank = comm.Get_rank()

    is_master = (not parallel) or (parallel and rank==0)
    start_time = time.time()
    nfe,last_log = 0,0

    if is_master:
      self.population = np.array([self.random_tree() for _ in range(self.popsize)])
      self.best_f = None
      self.best_P = None
      
      if log_frequency:
        snapshots = {'nfe': [], 'time': [], 'best_f': [], 'best_P': []}
    else:
      self.population = None


    while nfe < max_nfe:

      # evaluate objectives
      if not parallel:
        self.objectives = np.array([self.f(P) for P in self.population])
      else:
        if is_master:
          chunks = np.array_split(self.population, size)
        else:
          chunks = None

        local_Ps = comm.scatter(chunks, root=0)
        local_fs = [self.f(P) for P in local_Ps]
        objs = comm.gather(local_fs, root=0)
        comm.barrier()

        if is_master:
          self.objectives = np.concatenate(objs) # flatten list

      nfe += self.popsize

      if is_master:
        self.iterate()

        if log_frequency is not None and nfe >= last_log + log_frequency:
          elapsed = datetime.timedelta(seconds=time.time()-start_time).seconds

          if not self.multiobj:
            print('%d\t%s\t%0.3f\t%s' % (nfe, elapsed, self.best_f, self.best_P))
          else:
            print('# nfe = %d\n%s' % (nfe, self.best_f)) 
            print(self.best_f.shape)       
          snapshots['nfe'].append(nfe)
          snapshots['time'].append(elapsed)
          snapshots['best_f'].append(self.best_f)
          snapshots['best_P'].append(self.best_P)
          last_log = nfe
    
    if is_master and log_frequency:
      return snapshots
  

  def random_tree(self, terminal_ratio = 0.5):
    depth = np.random.randint(1, self.max_depth+1)
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

    T = PTree(L, self.feature_names)
    T.prune()
    return T


  def select_truncation(self, obj):
    return np.argsort(obj)[:self.mu]


  def crossover(self, P1, P2):
    P1,P2 = [copy.deepcopy(P) for P in (P1,P2)]
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


  def mutate(self, P, mutate_actions = True):
    P = copy.deepcopy(P)

    for item in P.L:
      if np.random.rand() < self.mut_prob:
        if item.is_feature:
          item.threshold = self.bounded_gaussian(item.threshold, self.feature_bounds[item.index])
        elif mutate_actions:
          if self.discrete_actions:
            item.value = str(np.random.choice(self.action_names))
          else:
            item.value = self.bounded_gaussian(item.value, self.action_bounds)

    return P


  def bounded_gaussian(self, x, bounds):
    # do mutation in normalized [0,1] to avoid sigma scaling issues
    lb,ub = bounds
    xnorm = (x-lb)/(ub-lb)
    x_trial = np.clip(xnorm + np.random.normal(0, scale=0.1), 0, 1)
        
    return lb + x_trial*(ub-lb)


  def dominates(self, a, b):
    # assumes minimization
    # a dominates b if it is <= in all objectives and < in at least one
    return (np.all(a <= b) and np.any(a < b))


  def same_box(self, a, b):
    if self.epsilons:
      a = a // self.epsilons
      b = b // self.epsilons
    return np.all(a == b)


  def binary_tournament(self, P, f):
    # select 1 parent from population P
    # (Luke Algorithm 99 p.138)
    i = np.random.randint(0,P.shape[0],2)
    a,b = f[i[0]], f[i[1]]
    if self.dominates(a,b):
      return i[0]
    elif self.dominates(b,a):
      return i[1]
    else:
      return i[0] if np.random.rand() < 0.5 else i[1]


  def archive_sort(self, A, fA, P, fP):
    A = np.hstack((A,P))
    fA = np.vstack((fA, fP))
    N = len(A)
    keep = np.ones(N, dtype=bool)

    for i in range(N):
      for j in range(i+1,N):
        if keep[j] and self.dominates(fA[i,:], fA[j,:]):# \
          keep[j] = False

        elif keep[i] and self.dominates(fA[j,:], fA[i,:]):
          keep[i] = False

        elif self.same_box(fA[i,:], fA[j,:]):
          keep[np.random.choice([i,j])] = False

    return (A[keep],fA[keep,:])

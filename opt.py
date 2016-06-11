import numpy as np
import time
from tree import *

# want to use Dave's "solution" and "problem" classes?

class PTreeOpt():

  def __init__(self, f, max_NFE, feature_bounds, action_bounds, 
               population_size = 100, mu = 15, max_depth = 4):

    self.f = f
    self.max_NFE = max_NFE
    self.num_features = len(feature_bounds)
    self.feature_bounds = feature_bounds
    self.action_bounds = action_bounds
    self.popsize = population_size
    self.mu = mu
    self.max_depth = max_depth
    self.nfe = 0


  def initialize(self):
    self.population = [self.random_tree() for _ in range(self.popsize)]
    self.objectives = [self.f(P) for P in self.population]


  def iterate(self):

    ix = np.argsort(self.objectives)[:self.mu]
    self.best_f = self.objectives[ix[0]]
    self.best_P = self.objectives[ix[0]]
    parents = self.population[ix]
    
    for i in range(self.popsize):
      pair = np.random.choice(parents, 2)
      child = self.crossover(*pair)
      self.population[i] = self.mutate(child)

    self.objectives = [self.f(P) for P in self.population]


  def run(self, condition):
            
    start_time = time.time()
    last_log = self.nfe
    
    self.initialize()

    while not self.nfe >= self.max_NFE:
      self.iterate()
      
      if self.log_frequency is not None and self.nfe >= last_log + self.log_frequency:
        pass # do something with these stats, append them
        # self.nfe,
        # datetime.timedelta(seconds=time.time()-start_time))
          

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
        L.append([np.random.uniform(*self.action_bounds[0])])

      else:
        x = np.random.choice(self.num_features)
        v = np.random.uniform(*self.feature_bounds[x])
        L.append([x,v])
        S += [current_depth+1]*2

    return PTree(L)

  def crossover(self, P1, P2):
    pass # do this with lists or trees?

  def mutate(self, P):
    pass



# here: mutation/crossover operators. Need ideas for these ... Koza?



'''
Created on 1 Dec 2018

@author: jhkwakkel
'''
from concurrent.futures import ProcessPoolExecutor

import datetime
import time
import numpy as np

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError as e:
    print(e)

class BaseExecutor(object):
    '''
    
    Parameters
    ----------
    algorithm : PTreeOpt instance
    kwargs : all kwargs will be passed on to
             concurrent.futures.ProcessPoolExecutor
    
    Attributes
    ----------
    algorithm : PTreeOpt instance
    
    '''
    
    def __init__(self, algorithm, **kwargs):
        
        self.algorithm = algorithm
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def map(self, population):
        raise NotImplementedError
        
    def run(self, max_nfe, log_frequency):
        start_time = time.time()
        nfe, last_log = 0, 0

        population = np.array(
            [self.algorithm.random_tree() for _ in range(self.algorithm.popsize)])
        self.algorithm.best_f = None
        self.algorithm.best_P = None
        self.algorithm.population = population

        if log_frequency:
            snapshots = {'nfe': [], 'time': [], 'best_f': [], 'best_P': []}
        else:
            self.algorithm.population = None

        while nfe < max_nfe:
            population = self.algorithm.population
            
            for member in population:
                member.clear_count() # reset action counts to zero

            # evaluate objectives            
            population, objectives = self.map(population)
            
            # TODO:: should not be attributes of algorithm
            self.algorithm.objectives = objectives
            self.algorithm.population = np.asarray(population)

            for member in population:
                member.normalize_count() # convert action count to percent

            nfe += self.algorithm.popsize

            self.algorithm.iterate()

            if log_frequency is not None and nfe >= last_log + log_frequency:
                elapsed = datetime.timedelta(
                    seconds=time.time() - start_time).seconds

                if not self.algorithm.multiobj:
                    print('%d\t%s\t%0.3f\t%s' %
                          (nfe, elapsed, self.algorithm.best_f, self.algorithm.best_P))
                else:
                    print('# nfe = %d\n%s' % (nfe, self.algorithm.best_f))
                    print(self.algorithm.best_f.shape)
                snapshots['nfe'].append(nfe)
                snapshots['time'].append(elapsed)
                snapshots['best_f'].append(self.algorithm.best_f)
                snapshots['best_P'].append(self.algorithm.best_P)
                last_log = nfe

        return snapshots


class MultiprocessingExecutor(BaseExecutor):
    '''
    
    Parameters
    ----------
    algorithm : PTreeOpt instance
    kwargs : all kwargs will be passed on to
             concurrent.futures.ProcessPoolExecutor
    
    Attributes
    ----------
    algorithm : PTreeOpt instance
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    '''
    def __init__(self, algorithm, **kwargs):
        super(MultiprocessingExecutor, self).__init__(algorithm)
        self.pool = ProcessPoolExecutor(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=True)
        return False
    
    def map(self, population):
        results = self.pool.map(self.algorithm.f, population)
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        
        return population, objectives
    

class MPIExecutor(BaseExecutor):
    '''
    
    Parameters
    ----------
    algorithm : PTreeOpt instance
    kwargs : all kwargs will be passed on to
             concurrent.futures.ProcessPoolExecutor
    
    Attributes
    ----------
    algorithm : PTreeOpt instance
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    '''
    
    def __init__(self, algorithm, **kwargs):
        super(MPIExecutor, self).__init__(algorithm)
        self.pool = MPIPoolExecutor(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=True)
        return False
    
    def map(self, population):
        results = self.pool.map(self.algorithm.f, population)
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        
        return population, objectives
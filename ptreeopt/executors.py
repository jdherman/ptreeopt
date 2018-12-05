'''
Created on 1 Dec 2018

@author: jhkwakkel
'''
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError as e:
    pass
    
class BaseExecutor(object):
    '''Base class for executor classes
    
    Parameters
    ----------
    kwargs : all kwargs will be passed on to the underlying executor
    
    '''
    
    def __init__(self, **kwargs):
        super(BaseExecutor, self).__init__()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def map(self, function, population):
        '''Map method to be implemeted by all subclasses
        
        Parameters
        ----------
        function : callable
        population  : collection
        
        Returns
        -------
        population
            collection with population members
        objectives
            collection with the scores for each population member
        
        '''
        
        raise NotImplementedError
        



class MultiprocessingExecutor(BaseExecutor):
    '''Executor for parallel execution using MultiProcessing
    
    Parameters
    ----------
    kwargs : all kwargs will be passed on to
             concurrent.futures.ProcessPoolExecutor
    
    Attributes
    ----------
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    '''
    def __init__(self, **kwargs):
        super(MultiprocessingExecutor, self).__init__()
        self.pool = ProcessPoolExecutor(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=True)
        return False
    
    def map(self, function, population):
        results = self.pool.map(function, population)
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        
        return population, objectives
    

class MPIExecutor(BaseExecutor):
    '''Exeuctor for parallel execution using MPI
    
    Parameters
    ----------
    kwargs : all kwargs will be passed on to
             mpi4py.futures.MPIPoolExecutor
    
    Attributes
    ----------
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    '''
    
    def __init__(self, **kwargs):
        super(MPIExecutor, self).__init__()
        self.pool = MPIPoolExecutor(**kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=True)
        return False
    
    def map(self, function, population):
        results = self.pool.map(function, population)
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        
        return population, objectives
    
class SequentialExecutor(BaseExecutor):
    '''Executor for sequential execution
    
    Parameters
    ----------
    algorithm : PTreeOpt instance
    
    Attributes
    ----------
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    '''
        
    def map(self, function, population):
        results = list(map(function, population))
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        
        return population, objectives
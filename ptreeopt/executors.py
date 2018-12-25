'''
Created on 1 Dec 2018

@author: jhkwakkel
'''

import multiprocessing
from multiprocessing import Pool, Queue

import logging
import logging.handlers
import threading

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


def initializer(queue, level):
    '''Helper function for initializing the logging for each of the
    sub processes.
    
    Parameters
    ----------
    queue : Multiprocessing.Queue instance
    level : int
            effective log level
    
    
    
    '''
    
    # Just the one handler neededs
    h = logging.handlers.QueueHandler(queue) 
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(level)
    
    # register a finalizer function that cleans up the process after
    # the pool is finished. 
    multiprocessing.util.Finalize(None, finalizer,
                                  args=(queue, ),
                                  exitpriority=10)


def finalizer(queue):
    queue.put(None) # ensures the reader thread exits cleanly


def listener_handler(queue):
    '''Helper function for reading log messages from the sub processes
    and re-log them using the logger of the main process
    
    Parameters
    ----------
    queue : multiprocessing.Queue instance
    
    '''

    # This is the listener thread top-level loop: wait for logging 
    # events (LogRecords)on the queue and handle them, quit when you
    # get a None for a LogRecord.    

    while True:
        try:
            record = queue.get()
            
            # We send None as a way to let the reader threat finalize
            # This None is put on the queue in the finalizer function
            # by each of the sub processes
            
            if record is None:  
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  
        except Exception:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)


class MultiprocessingExecutor(BaseExecutor):
    '''Executor for parallel execution using MultiProcessing
    
    Parameters
    ----------
    processes : int
    
    Attributes
    ----------
    pool : concurrent.futures.ProcessPoolExecutor instance
    
    
    TODO: I used a multiprocessing.Pool rather than
    concurrent.futures.ProcessPool because the initializer
    functions are available in python 3.6 for Pool, but requires
    3.7 for ProcessPool
    
    '''
    def __init__(self, processes=None):
        super(MultiprocessingExecutor, self).__init__()
        
        # we need to orchestrate the logging. We do this by setting
        # up a multiprocessing queue. The subprocess send their
        # log messages to the queue. We start a queue reader threat 
        # this reader takes log message from the queue and logs
        # them using the main process logger
        
        # we only want to propagate those log messages that we
        # are interested in
        level = logging.getLogger().getEffectiveLevel()
        
        queue = Queue(-1)
        self.pool = Pool(processes, initializer=initializer,
                         initargs=(queue, level))
        
        logthread = threading.Thread(target=listener_handler,
                                     args=(queue,), daemon=True)
        logthread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()
        return False
    
    def map(self, function, population):
        results = self.pool.map(function, population)
        population, objectives = list(zip(*results))
        
        objectives = np.asarray(objectives)
        return population, objectives


class MPIExecutor(BaseExecutor):
    '''Executor for parallel execution using MPI
    
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
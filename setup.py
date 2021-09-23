from setuptools import setup, find_packages

setup(name='ptreeopt',
      version='1.0.1',
      author='Jonathan Herman, Matteo Guilliani',
      author_email='jdherman@ucdavis.edu',
      url='https://github.com/jdherman/ptreeopt',
      description='A heuristic policy search model designed for the conrol of dynamic systems.',
      long_description = 'A heuristic policy search model designed for the control of dynamic systems. The model uses genetic programming to develop binary trees relating observed indicator variables to real-valued or discrete actions. A simulation model serves as optimization models objective function. Citation: Herman, J.D. and Giuliani, M. Policy tree optimization for threshold-based water resources management over multiple timescales, Environmental Modelling and Software, 99, 39-51, 2018.',
      packages=find_packages(),
      package_dir = {'': 'ptreeopt'},
      license = 'MIT',    
      )

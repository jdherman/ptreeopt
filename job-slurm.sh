#!/bin/bash -l
#SBATCH -n 100            # Total number of processors to request (32 cores per node)
#SBATCH -p high           # Queue name hi/med/lo
#SBATCH -t 1:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=jdherman@ucdavis.edu # address for email notification
#SBATCH --mail-type=ALL                  # email at Begin and End of job

# module load intel openmpi
# mpirun -n 50 python main-parallel-opt.py
module swap openmpi/1.6.5
mpirun -n 100 python -m cProfile -s time main-mpi.py > blah.txt

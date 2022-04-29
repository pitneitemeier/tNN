#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFI15ED
#
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72    # assign all the cores to that first task to make room for multithreading
#SBATCH --time=02:00:00
#SBATCH --output=out/TFI15edraven.out
#SBATCH --mail-type=none
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de
# set number of OMP threads *per process*
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


module load anaconda/3/2021.11

srun python3 ./hi.py

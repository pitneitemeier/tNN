#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFI4
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36 
#SBATCH --mem=250000
#SBATCH --time=0:05:00
#SBATCH --output=out/TFI4raven.out
#SBATCH --mail-type=none
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

./runner.sh

#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFIoptuna
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12500
#SBATCH --time=20:10:00
#SBATCH --output=out/TFI14raven_optuna1.out
#SBATCH --mail-type=none
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

module load intel/19.1.3 impi/2019.9 mkl/2020.4
module load cuda/11.2
module load anaconda/3/2021.11
python optuna_opt.py

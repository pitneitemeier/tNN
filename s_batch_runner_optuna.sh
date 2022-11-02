#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFIopt14
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=12500
#SBATCH --time=23:59:00
#SBATCH --output=out/TFI14_optuna_2.out
#SBATCH --mail-type=all
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

module load cuda/11.2
module load anaconda/3/2021.11
python optuna_opt_14.py

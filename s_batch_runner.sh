#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFI14_sr
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=250000
#
#SBATCH --time=10:00:00
#SBATCH --output=out/TFI14_sr.out
#SBATCH --mail-type=none
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

module load intel/19.1.3 impi/2019.9 mkl/2020.4
module load cuda/11.2
module load anaconda/3/2021.11
python TFI.py
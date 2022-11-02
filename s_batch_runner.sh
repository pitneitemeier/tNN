#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J TFI6
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --time=23:59:00
#SBATCH --output=out/TFI6_scaling.out
#SBATCH --mail-type=none
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

module load cuda/11.2
module load anaconda/3/2021.11
python TFI_scaling_6.py
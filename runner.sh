#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=thes1051
#SBATCH --cpus-per-task=48
#SBATCH --job-name=TFI8x
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:2
#SBATCH --time=0-1:00:00
#SBATCH --output=out/slurmTFI8x.out
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de

module load python/3.9.1
module load cuda/11.2
python3 model_runner.py
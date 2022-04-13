#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --job-name=TFI16ED
#SBATCH --partition=c18m
#SBATCH --time=0-7:00:00
#SBATCH --output=out/TFI15ED.out
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de
#SBATCH --mail-type=ALL


module load python
python3 ./hi.py

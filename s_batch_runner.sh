#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=thes1051
#SBATCH --cpus-per-task=48
#SBATCH --job-name=TFI10pulse
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:2
#SBATCH --time=0-5:00:00
#SBATCH --output=out/slurmTFI10pulse_FF_2.out
#SBATCH --mail-user=pit.neitemeier@rwth-aachen.de
#SBATCH --mail-type=ALL


module load python
module load cuda/11.4
python3 TFIz_pulse.py

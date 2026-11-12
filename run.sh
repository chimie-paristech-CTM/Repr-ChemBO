#!/bin/bash

set -e

module purge
module load anaconda-py3/2024.06
conda activate BO # CHANGE TO YOUR OWN ENV

python cache_pretrained_model.py

# required total cpu-cores indicated.
# here 1 cpu-core = 4GB memory on GENCI supercomputer.
# one can change the SBATCH script according to their machine.
sbatch scripts/submit_small_1.sh # 14,560
sbatch scripts/submit_small_2.sh # 8,320
sbatch scripts/submit_medium.sh # 5,760
sbatch scripts/submit_big.sh # 1,920
sbatch scripts/submit_random_search.sh # 380
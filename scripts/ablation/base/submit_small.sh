#!/bin/bash
#SBATCH --job-name=BO
#SBATCH -A qev@cpu
#SBATCH --array=0-7279
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t3
#SBATCH --output=logs/BO_%A_%a.out
#SBATCH --error=logs/BO_%A_%a.out

cd ${SLURM_SUBMIT_DIR}

module purge
module load anaconda-py3/2024.06

# source $(conda info --base)/etc/profile.d/conda.sh
conda activate BO

set -e
set -x

srun python main.py --config configs/ablation/base/config_small.json

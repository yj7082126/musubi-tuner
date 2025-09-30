#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --job-name=vistory
#SBATCH --output=logs/VS-%j.log

module load anaconda
module load gcc
module load cuda/cuda-12.4.0
conda activate myenv

echo "-------------------------------"
echo "Job Information:"
date
hostname
pwd
module list
echo "SLURM Nodes:  ${SLURM_NODELIST}"
echo "-------------------------------"

mkdir $HOME/tmp >> /dev/null  2>&1
export XDG_RUNTIME_DIR=$HOME/tmp

python vistory_testgen.py
date
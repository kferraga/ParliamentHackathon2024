#!/bin/bash
#SBATCH -A project_2010365
#SBATCH -J qDora
#SBATCH -o qDora.%j.out
#SBATCH -e qDora.%j.err
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niklas.oetken@stud.uni-bamberg.de

echo Starting at $(date) 
set -e 
# module load python-data/3.10-24.04

# set up wandb api
export WANDB_API_KEY=

./venv/bin/wandb login 
./venv/bin/python ./qDora.py


echo Finishing at $(date)

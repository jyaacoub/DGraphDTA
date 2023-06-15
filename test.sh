#!/bin/bash
##SBATCH --partition=gpu
#SBATCH --time=30
#SBATCH --job-name=torch_test
#SBATCH --output=/cluster/home/t122995uhn/projects/DGraphDTA/%x.out

#SBATCH --mem=500M
#SBATCH --cpus-per-task=1
#SBATCH --account=kumargroup_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

source .venv/bin/activate

python test.py 0 0 
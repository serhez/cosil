#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1

module restore cosil
source activate cosil

srun python gen_buffer.py ++seed=3

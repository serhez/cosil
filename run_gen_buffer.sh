#!/bin/bash -l
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module restore cosil
source activate cosil

srun python gen_buffer.py ++seed=3

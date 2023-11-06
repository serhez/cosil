#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

module restore cosil-new
source activate cosil

srun python gen_buffer.py ++seed=3

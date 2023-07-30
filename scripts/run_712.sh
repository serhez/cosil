#!/bin/bash -l
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

srun python train.py +experiment=25-07-2023/experiment/sail/om02-zsrew ++seed=712

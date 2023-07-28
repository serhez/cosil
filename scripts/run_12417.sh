#!/bin/bash -l
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

srun python train.py +experiment=25-07-2023/experiment/baseline/baseline ++seed=12417

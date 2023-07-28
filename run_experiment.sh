#!/bin/bash -l
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module restore cosil
source activate cosil

srun python train.py +experiment=11-05-2023/methods/cosil2-no-transfer-bo

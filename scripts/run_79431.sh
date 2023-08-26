#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

srun python train.py +experiment=final/experiment/sail/om00-oma05-zsrew-disctrans-pso ++seed=79431

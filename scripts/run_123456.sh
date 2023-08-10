#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

srun python train.py +experiment=final/pretrain/sail/om02-zsrew-disctrans-pos-pso ++seed=123456

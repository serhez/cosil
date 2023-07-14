#!/bin/bash -l
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

srun python train.py +experiment=13-07-2023/gail/om04-ss-repdemos-pso ++seed=214

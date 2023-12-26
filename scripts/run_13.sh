#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python train.py +experiment=scaledhumanoid/experiment/baseline/baseline-pso-zs ++seed=13

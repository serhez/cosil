#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python train.py +experiment=11-05-2023/methods/cosil-no-transfer-bo

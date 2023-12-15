#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python gen_buffer.py ++seed=3

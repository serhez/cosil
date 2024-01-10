#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python record_video.py +experiment=2seghalfcheetah/experiment/record_video ++seed=111

#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python record_video.py +experiment=humanoid/experiment/sail/om02-zsrew-disctrans-pso_short ++seed=111

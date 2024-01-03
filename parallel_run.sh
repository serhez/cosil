#!/bin/bash -l
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

case $SLURM_ARRAY_TASK_ID in
   0)  SEED=289 ;;
   1)  SEED=3 ;;
esac

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python train.py +experiment=humanoid/experiment/baseline/baseline-pso-zs ++seed=$SEED

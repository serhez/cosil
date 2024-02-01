#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sergio.hernandezgutierrez@aalto.fi
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

case $SLURM_ARRAY_TASK_ID in
   0)  SEED=111 ;;
   1)  SEED=123456 ;;
   2)  SEED=12417 ;;
   3)  SEED=13 ;;
   4)  SEED=214 ;;
esac

module restore cosil
source activate cosil

export MUJOCO_GL="egl"

xvfb-run python train.py +experiment=humanoid/experiment/baseline/baseline-pso-zs ++seed=$SEED

#!/bin/bash -l
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

module restore cosil-new
source activate cosil

for experiment in range-mean-1 range-mean-100 range-min-1 range-min-100 zscore-mean-1 zscore-mean-100 zscore-min-1 zscore-min-100
do
    echo "Running experiment: $experiment"
    srun python train.py +experiment=11-05-2023/normalization/"$experiment"
done

echo "Done!"

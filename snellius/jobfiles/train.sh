#!/bin/bash


#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=imagenet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=uva-fomo/snellius/logs/%A/slurm_output_%A.out
#SBATCH --error=uva-fomo/snellius/logs/%A/slurm_output_%A.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

python main.py \
    --epochs 10 \
    --num_channels 3 \
    --batch_size 256
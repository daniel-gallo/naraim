#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=/scratch-shared/fomo_logs/%A/stdout.txt
#SBATCH --error=/scratch-shared/fomo_logs/%A/stderr.txt

cp $0 /scratch-shared/fomo_logs/$SLURM_JOB_ID/script.sh
checkpoints_path="/scratch-shared/fomo_logs/$SLURM_JOB_ID/checkpoints"
tensorboard_path="/scratch-shared/fomo_logs/$SLURM_JOB_ID/tensorboard"

mkdir -p "$checkpoints_path"
mkdir -p "$tensorboard_path"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

python main.py \
    --profile \
    --max_num_iterations 4 \
    --warmup_steps 1 \
    --checkpoints_path "$checkpoints_path" \
    --tensorboard_path "$tensorboard_path"
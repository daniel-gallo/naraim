#!/bin/bash


#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=imagenet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/fomo_logs/logs/%A/slurm_output_%A.out
#SBATCH --error=/scratch-shared/fomo_logs/logs/%A/slurm_output_%A.err

cp $0 /scratch-shared/fomo_logs/$SLURM_JOB_ID

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

python main.py \
    --max_num_iterations 500000 \
    --batch_size 256 \
    --log_dir "/scratch-shared/fomo_logs/checkpoints"

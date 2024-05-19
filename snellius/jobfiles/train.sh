#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/scratch-shared/fomo_logs/%A/stdout.txt
#SBATCH --error=/scratch-shared/fomo_logs/%A/stderr.txt
####### --dependency=afterany:XXX

cp $0 /scratch-shared/fomo_logs/$SLURM_JOB_ID/script.sh
checkpoints_path="/scratch-shared/fomo_logs/$SLURM_JOB_ID/checkpoints"
tensorboard_path="/scratch-shared/fomo_logs/$SLURM_JOB_ID/tensorboard"

mkdir -p "$checkpoints_path"
mkdir -p "$tensorboard_path"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

previous_job_id=XXX
last_checkpoint=$(ls /scratch-shared/fomo_logs/$previous_job_id/checkpoints/ | sort -n | tail -1)
last_checkpoint_path=/scratch-shared/fomo_logs/$previous_job_id/checkpoints/$last_checkpoint
echo "Resuming from $last_checkpoint_path"

python main.py \
    --max_num_iterations 500000 \
    --checkpoints_path "$checkpoints_path" \
    --tensorboard_path "$tensorboard_path" \
    --native_resolutions \
    --checkpoint_path_to_load "$last_checkpoint_path"
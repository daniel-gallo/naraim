#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=imagenet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/fomo_logs/logs/%A/slurm_output_%A.out
#SBATCH --error=/scratch-shared/fomo_logs/logs/%A/slurm_output_%A.err

cp $0 /scratch-shared/fomo_logs/$SLURM_JOB_ID

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

# Run the program
python main.py \
    --max_num_iterations 500000 \
    --log_dir "/scratch-shared/fomo_logs/checkpoints"


log_dir="/scratch-shared/fomo_logs/checkpoints"
directory="/scratch-shared/fomo_logs/good_checkpoints"

################ Saving the checkpoints in a different way to ensure persistency ################
# Example: At the _fourth_ run, we will have:
# checkpoints/                 
#   - autoregressor/              
#       - state_{idx1}/           
#       - state_{idx2}/

# good_checkpoints/
#   - 1/
#       - autoregressor/              
#           - state_{idx1}/           
#           - state_{idx2}/
#   - 2/
#       - autoregressor/              
#           - state_{idx1}/           
#           - state_{idx2}/
#   - 3/
#       - autoregressor/              
#           - state_{idx1}/           
#           - state_{idx2}/
#   - 4/ *the new one
#       - autoregressor/              
#           - state_{idx1}/           
#           - state_{idx2}/

# Check if directory doesn't exist
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi

# Find the highest numbered folder from $directory
highest_number=$(find "$directory" -maxdepth 1 -type d -printf "%f\n" | grep -E '^[0-9]+$' | sort -n | tail -n 1)

# Increment the highest number to get the next number
next_number=$((highest_number + 1))

# Create the new directory
new_dir="${directory}/${next_number}"
mkdir "$new_dir"
echo "New directory created: $new_dir"

# Move the checkpoints to the new directory
mv "${log_dir}/"* $new_dir

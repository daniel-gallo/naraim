#!/bin/bash

if [[ "$VIRTUAL_ENV" == "" ]]
then
    source .venv/bin/activate
fi

python main.py \
    --dtype float32 \
    --num_layers 4 \
    --num_heads 4 \
    --embedding_dimension 128 \
    --hidden_dimension 256 \
    --max_num_iterations 20 \
    --warmup_steps 3 \
    --batch_size 32 \
    --eval_every_n_steps 10


directory="good_checkpoints"

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
mv checkpoints/* $new_dir



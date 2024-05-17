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
    --eval_every_n_steps 10 \
    --checkpoints_path checkpoints/tiny/classifier \
    --tensorboard_path tensorboard/tiny/classifier \
    --model_type classifier \
    --checkpoint_path_to_load checkpoints/tiny/autoregressor/step_20 \
    --num_categories 1000 \
    --load_only_params \

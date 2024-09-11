# NARAIM

Pre-training with native aspect ratio can be launched as follows:

```bash
python main.py \
    --dtype bfloat16 \
    --num_layers 12 \
    --num_heads 12 \
    --embedding_dimension 768 \
    --hidden_dimension 3072 \
    --max_num_iterations 500_000 \
    --warmup_steps 5_000 \
    --batch_size 512 \
    --checkpoints_path "./checkpoints/naraim-pre-training" \
    --tensorboard_path "./tensorboard/naraim-pre-training" \
    --train_transformations "random_horizontal_flip" "random_crop" "native_aspect_ratio_resize" \
    --validation_transformations "native_aspect_ratio_resize" \
    --dataset_path "path-to-tfrecords"
```

After this, we can use the pre-trained model to train a classifier:
```bash
python main.py \
    --dtype bfloat16 \
    --num_layers 12 \
    --num_heads 12 \
    --embedding_dimension 768 \
    --hidden_dimension 3072 \
    --max_num_iterations 50_000 \
    --warmup_steps 500 \
    --batch_size 512 \
    --checkpoints_path "./checkpoints/naraim-classifier" \
    --tensorboard_path "./tensorboard/naraim-classifier" \
    --model_type classifier \
    --checkpoint_path_to_load checkpoints/tiny/autoregressor/naraim-pre-training/step_500000 \
    --num_categories 1000 \
    --load_only_params \
    --train_transformations "random_horizontal_flip" "random_crop" "native_aspect_ratio_resize" \
    --validation_transformations "native_aspect_ratio_resize" \
    --dataset_path "path-to-tfrecords"
```

You can check `main.py` to see additional options, such as:
- `use_fractional_positional_encoding`, to replace the absolute positional encodings with fractional ones.
- `norm_pix_loss = False`, to use the un-normalized pre-training loss.
- Different transformations, such as `RandomResizedCrop` (for AIM training) or `AIMInference` (for AIM inference).
- `num_minibatches`, that can be increased if VRAM memory is scarce, as it will perform gradient accumulation. 
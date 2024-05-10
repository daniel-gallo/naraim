# uva-fomo

[![Status](https://github.com/jwpartyka/uva-fomo/actions/workflows/python.yml/badge.svg)](https://github.com/jwpartyka/uva-fomo/actions/workflows/python.yml)

## Dev instructions
1. Configure a virtual environment
2. Run `pip install pre-commit`
3. Run `pre-commit install`
4. Run `pre-commit run --all-files` before commit

## AIM paper
https://github.com/apple/ml-aim

## NaViT paper
https://github.com/kyegomez/NaViT

https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py

## MAE paper
https://github.com/facebookresearch/mae

## Checkpointing

### Checkpoint structure:

```json
ckpt = {
    "state": model_parameters,
    "metadata": {
        "model_type": model_type,
        "iteration": step,
        "lr_scheduler": {
            "init_value": 0.0,
            "peak_value": learning_rate,
            "decay_steps": max_num_iterations,
            "warmup_steps": warmup_steps,
        },
        "optimizer": {
            "adamw": {
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        },
    },
}
```

import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from jax import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ClassificationModel, PretrainingModel

# TODO: Adding lr scheduler / weight decay?
# TODO: Train/val/test split. Currently, there're only train and validation dataloaders

# TODO: Implementing a generic Trainer class


class Trainer:
    def __init__(
        self,
        model_type,
        dummy_batch,
        lr,
        seed,
        log_every_n_steps,
        norm_pix_loss,
        **model_hparams,
    ):
        super().__init__()
        self.model_type = model_type
        self.lr = lr
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.norm_pix_loss = norm_pix_loss
        self.log_every_n_steps = log_every_n_steps

        self.log_dir = os.path.join(str(Path.cwd()), f"checkpoints/{model_type}/")

        # Get empty model based on model_type
        self.model = (
            PretrainingModel(**model_hparams)
            if model_type == "autoregressor"
            else ClassificationModel(**model_hparams)
        )

        # Initialize model and logger
        self.init_model(dummy_batch, model_type)
        self.init_logger()

        # Jitting train and eval steps
        self.train_step = jax.jit(self.train_step)
        self.eval_step = jax.jit(self.eval_step)

        # Loss function
        self.get_loss = (
            self.loss_autoregressor
            if model_type == "autoregressor"
            else self.loss_classifier
        )

        # Metrics keys (for logging)
        self.metrics_keys = (
            ["mse"] if self.model_type == "autoregressor" else ["loss", "acc"]
        )

    def init_logger(self):
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def init_model(self, exmp_batch, model_type):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)

        (input_patches, patch_indices) = (
            (exmp_batch[0], exmp_batch[3])
            if model_type == "autoregressor"
            else (exmp_batch[0], exmp_batch[1])
        )

        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            input_patches,
            patch_indices,
            training=True,
        )["params"]
        self.state = None

    def init_optimizer(self):
        # TODO: lr_scheduler?

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(self.lr),
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer,
        )

    def loss_classifier(self, params, rng, batch, train):
        imgs, patch_indices, labels = batch
        rng, dropout_apply_rng = random.split(rng)
        logits = self.model.apply(
            {"params": params},
            imgs,
            patch_indices,
            training=train,
            rngs={"dropout": dropout_apply_rng},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()

        return loss, (rng, acc)

    def loss_autoregressor(self, params, rng, batch, train):
        input_patches, attn_mask, loss_mask, patch_indices, output_patches = batch

        # Normalize the target
        if self.norm_pix_loss:
            mean = jnp.mean(
                output_patches, axis=(-2, -1), keepdims=True
            )  # shape [bs, 1, 1]
            var = jnp.var(
                output_patches, axis=(-2, -1), keepdims=True
            )  # shape [bs, 1, 1]
            targets = (output_patches - mean) / (var + 1.0e-6) ** 0.5

        # Apply rng only for training
        if train:
            rng, dropout_apply_rng = random.split(rng)
            rngs = {"dropout": dropout_apply_rng}
        else:
            rngs = None

        preds = self.model.apply(
            {"params": params},
            input_patches,
            patch_indices=patch_indices,
            training=train,
            mask=attn_mask,
            rngs=rngs,
        )

        # TODO: Integrate positional embeddings
        # Pixel-wise MSE
        loss = (
            preds - targets
        ) ** 2  # shape = [bs, max_num_paches - 1, patch_size ** 2 * num_channels]

        # Apply mask on the loss so that gradients are computed
        # for the patches that are between the prefixed and padding patches
        loss = loss * loss_mask[:, :, None]
        loss = jnp.mean(loss)

        return loss, rng

    def train_step(self, state, rng, batch):
        def loss_fn(params):
            return self.get_loss(params, rng, batch, train=True)

        # Get output of loss function and gradients of the loss
        ## For autoregressor: output = (loss, rng)
        ## For classification: output = (loss, (rng, acc))
        output, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        # Update parameters and batch statistics
        state = state.apply_gradients(grads=grads)

        # Flatten the tuple
        if self.model_type == "classifier":
            output = [output]  # [(loss, (rng, acc))]
            output = [
                (loss, *aux_output) for loss, aux_output in output
            ]  # [(loss, rng, acc)]
            output = output[0]  # (loss, rng, acc)

        return state, output

    def eval_step(self, state, batch):
        # Return the mse for a single batch
        ## For autoregressor: output = (mse/loss, rng)
        ## For classification: output = (loss, rng, acc)
        output = self.get_loss(state.params, self.rng, batch, train=False)

        metric = output[0] if self.model_type == "autoregressor" else output[1][-1]

        return metric

    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and print avg loss and acc/mse
        metrics = defaultdict(list)

        for idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # aux_output:
            # - (loss, rng) for autoregressor
            # - (loss, rng, acc) for classification
            self.state, aux_output = self.train_step(self.state, self.rng, batch)

            self.rng = aux_output[1]  # rng

            for i, key in enumerate(self.metrics_keys):
                metrics[key].append(aux_output[2 * i])

            if idx > 1 and idx % self.log_every_n_steps == 0:
                step = (epoch - 1) * (len(train_loader) // 10) + idx / 10
                for key in self.metrics_keys:
                    self.logger.add_scalar(
                        f"{key}/train",
                        np.array(jax.device_get(metrics)[key]).mean(),
                        step,
                    )

        metrics = jax.device_get(metrics)
        metrics = {key: np.array(metric).mean() for key, metric in metrics.items()}

        return metrics

    def train_model(self, train_loader, val_loader, num_epochs):
        # Train model for defined number of epochs
        self.init_optimizer()
        # Track best eval metric
        best_eval = float("-inf") if self.model_type == "autoregressor" else 0.0
        hparams_dict = {"learning_rate": self.lr, "seed": self.seed}

        metric_to_eval = "mse" if "mse" in self.metrics_keys else "acc"
        best_metrics = {
            f"Best_{metric_to_eval}/train": None,
            f"Best_{metric_to_eval}/val": None,
        }

        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            train_metrics = self.train_epoch(train_loader, epoch_idx)
            eval_metric = self.eval_model(val_loader)

            # TODO: Make this nicer
            if metric_to_eval == "mse":
                eval_metric = -eval_metric

            if eval_metric >= best_eval:
                best_eval = eval_metric
                self.save_model(step=epoch_idx)
                best_metrics[f"Best_{metric_to_eval}/train"] = train_metrics[
                    metric_to_eval
                ]

                best_metrics[f"Best_{metric_to_eval}/val"] = (
                    -eval_metric if metric_to_eval == "mse" else eval_metric
                )

            # TODO: Now there is duplicate code
            if metric_to_eval == "mse":
                eval_metric = -eval_metric

            # Log the metric
            self.logger.add_scalar(f"{metric_to_eval}/val", eval_metric, epoch_idx)

        self.logger.add_hparams(hparams_dict, best_metrics)
        self.logger.flush()
        self.logger.close()

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg mse or acc
        total_val, count = 0.0, 0

        for batch in data_loader:
            val = self.eval_step(self.state, batch)
            total_val += val * batch[0].shape[0]
            count += batch[0].shape[0]

        res = (total_val / count).item()

        return res

    def save_model(self, step):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state.params, step=step, overwrite=True
        )

    def load_model(self):
        # Load model
        params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx
            if self.state
            else optax.adamw(self.lr),  # Default optimizer
        )

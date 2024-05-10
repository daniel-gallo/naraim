import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from jax import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ClassificationModel, PretrainingModel

# TODO: Adding lr scheduler / weight decay?


class Trainer:
    def __init__(
        self,
        model_type,
        dummy_batch,
        lr,
        seed,
        log_every_n_steps,
        eval_every_n_steps,
        log_dir,
        norm_pix_loss,
        decay_steps,
        model_hparams,
    ):
        super().__init__()
        self.model_type = model_type
        self.lr = lr
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.norm_pix_loss = norm_pix_loss
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.decay_steps = decay_steps

        self.log_dir = str((Path(log_dir) / model_type).absolute())

        # Get empty model based on model_type
        self.model = (
            PretrainingModel(**model_hparams)
            if model_type == "autoregressor"
            else ClassificationModel(**model_hparams)
        )

        # Checkpointing
        ## Create a logging directory if it has not been created yet
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        ## Set up the checkpointer
        ## TODO: Change the enable_async_checkpointing = True when working on multiple GPUs
        options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=2,
            step_prefix="state",
            enable_async_checkpointing=False,
        )
        ## TODO: This will delete the previous log directory (erase_and_create_empty)
        ## Should we move them automatically to another directory before a new run?
        self.checkpoint_manager = ocp.CheckpointManager(
            ocp.test_utils.erase_and_create_empty(self.log_dir),
            options=options,
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

        image, image_coords, label, attention_matrix, loss_mask = exmp_batch

        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            image,
            image_coords,
            training=True,
        )["params"]
        self.state = None

    def init_optimizer(self):
        self.lr_schedule = optax.cosine_decay_schedule(
            init_value=self.lr, decay_steps=self.decay_steps
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(self.lr_schedule),
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer,
        )

    def loss_classifier(self, params, rng, batch, train):
        image, image_coords, labels, attention_matrix, loss_mask = batch
        rng, dropout_apply_rng = random.split(rng)
        logits = self.model.apply(
            {"params": params},
            image,
            image_coords,
            training=train,
            rngs={"dropout": dropout_apply_rng},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()

        return loss, (rng, acc)

    def loss_autoregressor(self, params, rng, batch, train):
        patches, patch_indices, labels, attention_matrices, loss_masks = batch

        # Normalize the target
        if self.norm_pix_loss:
            mean = jnp.mean(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
            var = jnp.var(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
            targets = (patches - mean) / (var + 1.0e-6) ** 0.5
        else:
            targets = patches

        # Apply rng only for training
        if train:
            rng, dropout_apply_rng = random.split(rng)
            rngs = {"dropout": dropout_apply_rng}
        else:
            rngs = None

        preds = self.model.apply(
            {"params": params},
            patches,
            patch_indices=patch_indices,
            training=train,
            mask=attention_matrices,
            rngs=rngs,
        )

        # We predict the next patch, so we need to slice the tensors
        loss = (
            preds[:, :-1, :] - targets[:, 1:, :]
        ) ** 2  # shape = [bs, max_num_paches - 1, patch_size ** 2 * num_channels]

        # Apply mask on the loss so that gradients are computed
        # for the patches that are between the prefixed and padding patches
        loss = loss * loss_masks[:, :-1, None]
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

    def train_model(self, train_loader, val_loader, max_num_iterations):
        # Train model for defined number of epochs
        self.init_optimizer()
        # Track best eval metric
        best_eval = float("-inf") if self.model_type == "autoregressor" else 0.0
        hparams_dict = {"learning_rate": self.lr_schedule(0), "seed": self.seed}

        metric_to_eval = "mse" if "mse" in self.metrics_keys else "acc"
        best_metrics = {
            f"Best_{metric_to_eval}/train": None,
            f"Best_{metric_to_eval}/val": None,
        }

        step = 0

        train_metrics = defaultdict(list)

        for idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # Finish training after the max_num_iterations steps
            if idx >= max_num_iterations:
                break

            # aux_output:
            # - (loss, rng) for autoregressor
            # - (loss, rng, acc) for classification
            self.state, aux_output = self.train_step(self.state, self.rng, batch)
            self.rng = aux_output[1]  # rng
            for i, key in enumerate(self.metrics_keys):
                train_metrics[key].append(aux_output[2 * i])

            # Logging
            step += 1
            if idx > 1 and idx % self.log_every_n_steps == 0:
                for key in self.metrics_keys:
                    self.logger.add_scalar(
                        f"{key}/train",
                        np.array(jax.device_get(train_metrics)[key]).mean(),
                        step,
                    )

            # Do evaluation once eval_steps
            if idx > 1 and idx % self.eval_every_n_steps == 0:
                train_metrics = jax.device_get(train_metrics)
                train_metrics = {
                    key: np.array(metric).mean()
                    for key, metric in train_metrics.items()
                }

                eval_metric = self.eval_model(val_loader)

                # TODO: Make this nicer
                if metric_to_eval == "mse":
                    eval_metric = -eval_metric

                if eval_metric >= best_eval:
                    best_eval = eval_metric
                    self.save_model(step=idx)
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
                self.logger.add_scalar(f"{metric_to_eval}/val", eval_metric, idx)
                self.logger.add_hparams(hparams_dict, best_metrics)

                # Reset train_metrics dictionary????
                # train_metrics = defaultdict(list)

        # self.logger.add_hparams(hparams_dict, best_metrics)
        self.logger.flush()
        self.logger.close()

        # for epoch_idx in tqdm(range(1, num_iterations + 1)):
        #     train_metrics = self.train_epoch(train_loader, epoch_idx)
        #     eval_metric = self.eval_model(val_loader)

        #     # TODO: Make this nicer
        #     if metric_to_eval == "mse":
        #         eval_metric = -eval_metric

        #     if eval_metric >= best_eval:
        #         best_eval = eval_metric
        #         self.save_model(step=epoch_idx)
        #         best_metrics[f"Best_{metric_to_eval}/train"] = train_metrics[
        #             metric_to_eval
        #         ]

        #         best_metrics[f"Best_{metric_to_eval}/val"] = (
        #             -eval_metric if metric_to_eval == "mse" else eval_metric
        #         )

        #     # TODO: Now there is duplicate code
        #     if metric_to_eval == "mse":
        #         eval_metric = -eval_metric

        #     # Log the metric
        #     self.logger.add_scalar(f"{metric_to_eval}/val", eval_metric, epoch_idx)

        # self.logger.add_hparams(hparams_dict, best_metrics)
        # self.logger.flush()
        # self.logger.close()

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
        self.checkpoint_manager.save(
            step,
            self.state.params,
            args=ocp.args.StandardSave(self.state.params),
            force=True,
        )

    def load_model(self, step):
        # Load model from a certain training iteration
        params = self.checkpoint_manager.restore(step, items=None)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx
            if self.state
            else optax.adamw(self.lr),  # Default optimizer
        )

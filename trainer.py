from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from jax import random
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset import prefetch
from model import ClassificationModel, PretrainingModel


class Trainer:
    def __init__(
        self,
        model_type,
        dummy_batch,
        lr,
        beta2,
        weight_decay,
        seed,
        log_every_n_steps,
        eval_every_n_steps,
        checkpoints_path,
        tensorboard_path,
        checkpoint_path_to_load,
        load_only_params,
        norm_pix_loss,
        max_num_iterations,
        warmup_steps,
        model_hparams,
        profile,
    ):
        super().__init__()
        self.model_type = model_type
        self.lr = lr
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.norm_pix_loss = norm_pix_loss
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.max_num_iterations = max_num_iterations
        self.warmup_steps = warmup_steps
        self.tensorboard_path = tensorboard_path
        self.checkpoints_path = Path(checkpoints_path).absolute()
        self.profile = profile

        # Get empty model based on model_type
        self.model = (
            PretrainingModel(**model_hparams)
            if model_type == "autoregressor"
            else ClassificationModel(**model_hparams)
        )

        # Initialize model, logger and optimizer
        self.init_model(dummy_batch, model_type)
        self.init_logger()
        self.init_optimizer()
        if checkpoint_path_to_load:
            self.load_checkpoint(checkpoint_path_to_load, model_type, load_only_params)

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
        self.logger = SummaryWriter(log_dir=self.tensorboard_path)

    def init_model(self, exmp_batch, model_type):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)

        image, image_coords, label, attention_matrix, loss_mask = exmp_batch

        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            image,
            image_coords,
            training=True,
        )["params"]
        param_count = sum(x.size for x in jax.tree_leaves(self.init_params))
        print(f"Number of parameters: {param_count}")

    def init_optimizer(self):
        # If we do not have any checkpoint to load, then create a new state
        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            decay_steps=self.max_num_iterations,
            warmup_steps=self.warmup_steps,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(
                self.lr_schedule, b2=self.beta2, weight_decay=self.weight_decay
            ),
        )

        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params,
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
        # Track best eval metric
        best_eval = float("-inf") if self.model_type == "autoregressor" else 0.0
        hparams_dict = {"learning_rate": self.lr, "seed": self.seed}

        metric_to_eval = "mse" if "mse" in self.metrics_keys else "acc"
        best_metrics = {
            f"Best_{metric_to_eval}/val": None,
        }

        train_metrics = defaultdict(list)

        if self.profile:
            jax.profiler.start_trace(self.tensorboard_path)

        # If we load the checkpoint, the first step will not be zero
        first_step = self.state.step
        for idx, batch in zip(
            trange(
                first_step,
                max_num_iterations,
                initial=first_step,
                total=max_num_iterations,
                desc="Training",
            ),
            train_loader,
        ):
            if self.profile:
                context_manager = jax.profiler.StepTraceAnnotation(
                    "train_step",
                    step_num=idx + 1,
                )
            else:
                context_manager = nullcontext()
            with context_manager:
                # aux_output:
                # - (loss, rng) for autoregressor
                # - (loss, rng, acc) for classification
                self.state, aux_output = self.train_step(self.state, self.rng, batch)

            self.rng = aux_output[1]  # rng
            for i, key in enumerate(self.metrics_keys):
                train_metrics[key].append(aux_output[2 * i])

            # Logging
            if idx > first_step and (idx + 1) % self.log_every_n_steps == 0:
                for key in self.metrics_keys:
                    self.logger.add_scalar(
                        f"{key}/train",
                        np.array(jax.device_get(train_metrics)[key]).mean(),
                        idx,
                    )
                # Reinitialize the train_metrics
                train_metrics = defaultdict(list)

            # Do evaluation once eval_steps
            if idx > first_step and (idx + 1) % self.eval_every_n_steps == 0:
                self.save_checkpoint(step=idx + 1)
                # Evaluate the model and return the eval metrics
                ckpt_val_loader = val_loader.save()
                eval_metric = self.eval_model(prefetch(val_loader))
                val_loader.restore(ckpt_val_loader)

                if metric_to_eval == "mse":
                    eval_metric = -eval_metric

                if eval_metric >= best_eval:
                    best_eval = eval_metric

                    best_metrics[f"Best_{metric_to_eval}/val"] = (
                        -eval_metric if metric_to_eval == "mse" else eval_metric
                    )

                if metric_to_eval == "mse":
                    eval_metric = -eval_metric

                # Log the metric
                self.logger.add_scalar(f"{metric_to_eval}/val", eval_metric, idx)

                hparams_dict["learning_rate"] = float(self.lr_schedule(idx))
                self.logger.add_hparams(hparams_dict, best_metrics)

                # Flush the logger
                self.logger.flush()

        if self.profile:
            train_metrics[metric_to_eval][-1].block_until_ready()
            jax.profiler.stop_trace()

        self.logger.close()

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg mse or acc
        total_val, count = 0.0, 0

        for batch in tqdm(data_loader, desc="Validation", leave=False):
            val = self.eval_step(self.state, batch)
            total_val += val * batch[0].shape[0]
            count += batch[0].shape[0]

        res = (total_val / count).item()

        return res

    def save_checkpoint(self, step):
        checkpointer = AsyncCheckpointer(PyTreeCheckpointHandler())
        checkpointer.save(self.checkpoints_path / f"step_{step}", self.state)

    def load_checkpoint(
        self, checkpoint_path_to_load, model_type, load_only_params=False
    ):
        # Restore the lastest checkpoint (the best saved model)
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(Path(checkpoint_path_to_load).absolute())

        if load_only_params:
            assert model_type == "classifier"

            if "PretrainingHead_0" in restored["params"]:
                restored["params"].pop("PretrainingHead_0")
                restored["params"]["ClassificationHead_0"] = self.state.params[
                    "ClassificationHead_0"
                ]

            self.state = self.state.replace(
                params=restored["params"],
            )

        else:
            restored_opt_state = jax.tree_unflatten(
                jax.tree_structure(self.state.opt_state),
                jax.tree_leaves(restored["opt_state"]),
            )

            self.state = self.state.replace(
                params=restored["params"],
                step=int(restored["step"]),
                opt_state=restored_opt_state,
            )

import functools
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import traverse_util
from flax.training import train_state
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset import prefetch
from model import ClassificationModel, PretrainingModel


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def init_model(
    rng: jax.random.PRNGKey, exmp_batch: jax.Array, model: nn.Module, optimizer
) -> (train_state.TrainState, jax.random.PRNGKey):
    image, image_coords, label, attention_matrix, loss_mask = exmp_batch
    rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
    params = model.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        image.squeeze(),
        image_coords.squeeze(),
        training=True,
    )["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    return state, rng


def loss_classifier(params, apply_fn, batch, dropout_rng, train):
    image, image_coords, labels, attention_matrix, loss_mask = batch
    logits = apply_fn(
        {"params": params},
        image,
        image_coords,
        training=train,
        rngs={"dropout": dropout_rng},
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (logits.argmax(axis=-1) == labels).mean()

    batch_size = image.shape[0]
    step_metrics = {"loss": (loss.sum(), batch_size), "accuracy": (acc, batch_size)}

    return loss, step_metrics


@partial(jax.jit, static_argnames=["apply_fn", "mesh", "norm_pix_loss", "train"])
def loss_autoregressor_dp(
    params, batch, apply_fn, mesh, norm_pix_loss, train, dropout_rng
):
    @partial(shard_map, mesh=mesh, in_specs=P("data"), out_specs=P())
    def loss_autoregressor_spmd(local_batch):
        patches, patch_indices, labels, attention_matrices, loss_masks = local_batch
        if norm_pix_loss:
            mean = jnp.mean(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
            var = jnp.var(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
            targets = (patches - mean) / (var + 1.0e-6) ** 0.5
        else:
            targets = patches

        if train:
            rngs = {"dropout": dropout_rng}
        else:
            rngs = None

        preds = apply_fn(
            {"params": params},
            patches,
            patch_indices=patch_indices,
            training=train,
            mask=attention_matrices,
            rngs=rngs,
        )

        # We predict the next patch, so we need to slice the tensors
        local_loss = (
            preds[:, :-1, :] - targets[:, 1:, :]
        ) ** 2  # shape = [bs, max_num_paches - 1, patch_size ** 2 * num_channels]

        # Apply mask on the loss so that gradients are computed
        # for the patches that are between the prefixed and padding patches
        local_loss = local_loss * loss_masks[:, :-1, None]
        local_loss = jnp.mean(local_loss)

        # batch_size = batch[0].shape[0]
        # step_metrics = {"mse": (loss, batch_size)}
        return jax.lax.pmean(local_loss, "data")

    return loss_autoregressor_spmd(batch)


@partial(
    jax.jit, static_argnames=["apply_fn", "mesh", "norm_pix_loss", "train", "loss_fn"]
)
def train_step(
    state: train_state.TrainState,
    batch,
    apply_fn,
    mesh,
    norm_pix_loss,
    train,
    rng,
    loss_fn,
):
    rng, step_rng = jax.random.split(rng)
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, batch, apply_fn, mesh, norm_pix_loss, train, step_rng
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, rng


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
        freeze_backbone,
        norm_pix_loss,
        max_num_iterations,
        warmup_steps,
        model_hparams,
        profile,
        grad_clip_norm,
        lr_end_value,
        n_images_to_visualize,
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
        self.lr_end_value = lr_end_value
        self.grad_clip_norm = grad_clip_norm
        self.n_images_to_visualize = n_images_to_visualize

        device_array = np.array(jax.devices())
        self.mesh = Mesh(device_array, ("data",))

        # Get empty model based on model_type
        self.model = (
            PretrainingModel(**model_hparams)
            if model_type == "autoregressor"
            else ClassificationModel(**model_hparams)
        )

        # Initialize model, logger and optimizer
        self.init_optimizer()
        self.state, self.rng = init_model(
            self.rng, dummy_batch, self.model, self.optimizer
        )
        self.init_logger()
        if checkpoint_path_to_load:
            self.load_checkpoint(
                checkpoint_path_to_load, model_type, load_only_params, freeze_backbone
            )

        # Loss function
        # self.get_loss = (
        #     functools.partial(loss_autoregressor, norm_pix_loss=self.norm_pix_loss)
        #     if model_type == "autoregressor"
        #     else loss_classifier
        # )
        self.get_loss = loss_autoregressor_dp

        # Metrics keys (for logging)
        self.metrics_keys = (
            ["mse"] if self.model_type == "autoregressor" else ["loss", "acc"]
        )

    def init_logger(self):
        self.logger = SummaryWriter(log_dir=self.tensorboard_path)

    def init_optimizer(self):
        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            decay_steps=self.max_num_iterations,
            warmup_steps=self.warmup_steps,
            end_value=self.lr_end_value,
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),  # Clip gradients at norm 1
            optax.adamw(
                self.lr_schedule, b2=self.beta2, weight_decay=self.weight_decay
            ),
        )

    # def eval_step(self, state, rng, batch):
    #     # Return the mse for a single batch
    #     ## For autoregressor: output = (mse/loss, rng)
    #     ## For classification: output = (loss, rng, acc)
    #     (_, rng, metrics) = self.get_loss(state.params, rng, batch, train=False)
    #
    #     # metric = output[0] if self.model_type == "autoregressor" else output[1][-1]
    #
    #     return rng, metrics

    def train_model(self, train_loader, val_loader, max_num_iterations):
        # Track best eval metric
        best_eval = float("-inf") if self.model_type == "autoregressor" else 0.0

        metric_to_eval = "mse" if "mse" in self.metrics_keys else "acc"

        train_metrics = None

        if self.profile:
            jax.profiler.start_trace(self.tensorboard_path)

        # If we load the checkpoint, the first step will not be zero
        first_step = int(self.state.step)
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
                self.state, self.rng = train_step(
                    self.state,
                    batch,
                    self.state.apply_fn,
                    self.mesh,
                    self.norm_pix_loss,
                    True,
                    self.rng,
                    self.get_loss,
                )
            # print(jax.device_get(train_metrics['mse']).mean())
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

                if metric_to_eval == "mse":
                    eval_metric = -eval_metric

                # Log the metric
                self.logger.add_scalar(f"{metric_to_eval}/val", eval_metric, idx)

                # Log the learning rate
                self.logger.add_scalar(
                    "Learning rate", float(self.lr_schedule(idx)), idx
                )

                # Flush the logger
                self.logger.flush()

        if self.profile:
            train_metrics[metric_to_eval][-1].block_until_ready()
            jax.profiler.stop_trace()

        self.logger.close()

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg mse or acc
        total_val, count = 0.0, 0

        batches_to_skip = np.random.randint(1, 50)

        for idx, batch in enumerate(tqdm(data_loader, desc="Validation", leave=False)):
            if self.model_type == "autoregressor":
                # Always plot the same few images for easy comparison across training stages
                if idx == 0:
                    patches, patch_indices, _, attention_matrices, _ = batch
                    self.visualize_pretraining_output(
                        patches,
                        patch_indices,
                        attention_matrices,
                        plot_random_images=False,
                    )

                # Also plot a few random images for a more representative sample
                if idx == batches_to_skip:
                    patches, patch_indices, _, attention_matrices, _ = batch
                    self.visualize_pretraining_output(
                        patches,
                        patch_indices,
                        attention_matrices,
                        plot_random_images=True,
                    )

            val = self.eval_step(self.state, batch)
            total_val += val * batch[0].shape[0]
            count += batch[0].shape[0]

        res = (total_val / count).item()

        return res

    def save_checkpoint(self, step):
        checkpointer = AsyncCheckpointer(PyTreeCheckpointHandler())
        checkpointer.save(self.checkpoints_path / f"step_{step}", self.state)

    def load_checkpoint(
        self,
        checkpoint_path_to_load,
        model_type,
        load_only_params=False,
        freeze_backbone=False,
    ):
        # Restore the lastest checkpoint (the best saved model)
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(Path(checkpoint_path_to_load).absolute())

        if load_only_params:
            print("Loading only parameters")
            assert model_type == "classifier"

            if "PretrainingHead_0" in restored["params"]:
                restored["params"].pop("PretrainingHead_0")
                restored["params"]["ClassificationHead_0"] = self.state.params[
                    "ClassificationHead_0"
                ]

            if freeze_backbone:
                print("Freezing the backbone")

                partition_optimizers = {
                    "trainable": self.optimizer,
                    "frozen": optax.set_to_zero(),
                }

                param_partitions = traverse_util.path_aware_map(
                    lambda path, _: "frozen"
                    if "ClassificationHead_0" not in path
                    else "trainable",
                    restored["params"],
                )
                tx = optax.multi_transform(partition_optimizers, param_partitions)

                self.state = train_state.TrainState.create(
                    apply_fn=self.model.apply,
                    params=restored["params"],
                    tx=tx,
                )
            else:
                self.state = self.state.replace(
                    params=restored["params"],
                )
        else:
            print("Loading full train state")
            restored_opt_state = jax.tree_unflatten(
                jax.tree_structure(self.state.opt_state),
                jax.tree_leaves(restored["opt_state"]),
            )

            self.state = self.state.replace(
                params=restored["params"],
                step=int(restored["step"]),
                opt_state=restored_opt_state,
            )

    def visualize_pretraining_output(
        self, patches, patch_indices, attention_matrices, plot_random_images
    ):
        if plot_random_images:
            transposition_indices = np.random.permutation(patches.shape[0])
            patches = patches[transposition_indices]
            patch_indices = patch_indices[transposition_indices]
            attention_matrices = attention_matrices[transposition_indices]

        mean = jnp.mean(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
        var = jnp.var(patches, axis=-1, keepdims=True)  # shape [bs, patches, 1]
        targets = (patches - mean) / (var + 1.0e-6) ** 0.5
        preds = self.model.apply(
            {"params": self.state.params},
            patches,
            patch_indices=patch_indices,
            training=False,
            mask=attention_matrices,
            rngs=None,
        )

        for i in range(self.n_images_to_visualize):
            max_row = patch_indices[i][:, 0].max()
            max_col = patch_indices[i][:, 1].max()
            tgt_norm = np.zeros((3, (max_row + 1) * 14, (max_col + 1) * 14))
            pred_norm = np.zeros((3, (max_row + 1) * 14, (max_col + 1) * 14))
            tgt_unnorm = np.zeros((3, (max_row + 1) * 14, (max_col + 1) * 14))
            pred_unnorm = np.zeros((3, (max_row + 1) * 14, (max_col + 1) * 14))

            for idx, (row, col) in enumerate(patch_indices[i]):
                if idx > 0 and row == 0 and col == 0:  # skip padded tokens
                    break

                tgt_norm[:, row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14] = (
                    rearrange(targets[i][idx], "(h w c) -> c h w", h=14, w=14, c=3)
                )
                pred_norm[:, row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14] = (
                    rearrange(preds[i][idx], "(h w c) -> c h w", h=14, w=14, c=3)
                )

            for idx, (row, col) in enumerate(patch_indices[i]):
                if idx > 0 and row == 0 and col == 0:  # skip padded tokens
                    break

                # invert the normalization
                tgt_unnorm[:, row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14] = (
                    rearrange(targets[i][idx], "(h w c) -> c h w", h=14, w=14, c=3)
                    * (var[i][idx] + 1.0e-6) ** 0.5
                ) + mean[i][idx]
                pred_unnorm[:, row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14] = (
                    rearrange(preds[i][idx], "(h w c) -> c h w", h=14, w=14, c=3)
                    * (var[i][idx] + 1.0e-6) ** 0.5
                ) + mean[i][idx]

            # Tensorboard only accepts uint8 images
            tgt_norm = (tgt_norm.clip(0, 1) * 255).astype(np.uint8)
            tgt_unnorm = (tgt_unnorm.clip(0, 1) * 255).astype(np.uint8)
            pred_norm = (pred_norm.clip(0, 1) * 255).astype(np.uint8)
            pred_unnorm = (pred_unnorm.clip(0, 1) * 255).astype(np.uint8)

            if plot_random_images:
                self.logger.add_images(
                    f"Random image {i+1}",
                    np.array([tgt_norm, tgt_unnorm, pred_norm, pred_unnorm]),
                    global_step=self.state.step,
                )
            else:
                self.logger.add_images(
                    f"Static image {i+1}",
                    np.array([tgt_norm, tgt_unnorm, pred_norm, pred_unnorm]),
                    global_step=self.state.step,
                )
        return

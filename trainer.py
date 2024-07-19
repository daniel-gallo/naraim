from contextlib import nullcontext
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import traverse_util
from flax.struct import dataclass
from flax.training import train_state
from jax import random
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from common import warmup_exponential_decay_cooldown_scheduler
from dataset import prefetch
from model import ClassificationModel, PretrainingModel
from model.classification import NoTransformerClassificationModel


def translate(target, source):
    """
    If some remats are added / removed from the mode, the checkpoints will not be compatible.
    We can solve this by adding / removing the "Checkpoint" prefix as needed.
    """
    normalized_to_target = {}
    for target_path in traverse_util.flatten_dict(target):
        normalized_path = tuple(key.replace("Checkpoint", "") for key in target_path)
        normalized_to_target[normalized_path] = target_path

    translated_params = {}
    for source_path, value in traverse_util.flatten_dict(source).items():
        normalized_path = tuple(key.replace("Checkpoint", "") for key in source_path)
        d1_path = normalized_to_target[normalized_path]

        translated_params[d1_path] = value

    return traverse_util.unflatten_dict(translated_params)


@dataclass
class Batch:
    patches: jax.Array
    patch_coordinates: jax.Array
    labels: jax.Array
    attention_masks: jax.Array
    loss_masks: jax.Array


class Trainer:
    def __init__(
        self,
        model_type,
        dummy_batch,
        lr,
        decay_rate,
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
        cooldown_steps,
        model_hparams,
        profile,
        grad_clip_norm,
        lr_end_value,
        n_images_to_visualize,
        num_minibatches,
        lr_schedule_type,
    ):
        super().__init__()
        self.model_type = model_type
        self.lr = lr
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.norm_pix_loss = norm_pix_loss
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.max_num_iterations = max_num_iterations
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.tensorboard_path = tensorboard_path
        self.checkpoints_path = Path(checkpoints_path).absolute()
        self.profile = profile
        self.lr_end_value = lr_end_value
        self.grad_clip_norm = grad_clip_norm
        self.n_images_to_visualize = n_images_to_visualize
        self.num_minibatches = num_minibatches
        self.lr_schedule_type = lr_schedule_type

        # Get empty model based on model_type
        if model_type == "autoregressor":
            self.model = PretrainingModel(**model_hparams)
        elif model_type == "no_transformer_classifier":
            print("Look mom, no transformer!")
            self.model = NoTransformerClassificationModel(**model_hparams)
        else:
            self.model = ClassificationModel(**model_hparams)

        # Initialize model, logger and optimizer
        self.init_model(dummy_batch)
        self.init_logger()
        self.init_optimizer(freeze_backbone=freeze_backbone)
        if checkpoint_path_to_load:
            self.load_checkpoint(checkpoint_path_to_load, model_type, load_only_params)

        # Async checkpointer
        # TODO: Reimplement checkpointing
        self.checkpointer = AsyncCheckpointer(PyTreeCheckpointHandler())

        # Jitting train and eval steps
        self.train_step = jax.jit(
            self.train_step, static_argnames=("loss_fn", "num_minibatches")
        )
        self.eval_step = jax.jit(self.eval_step, static_argnames="loss_fn")

    def init_logger(self):
        self.logger = SummaryWriter(log_dir=self.tensorboard_path)

    def init_model(self, exmp_batch):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)

        image, image_coords, label, attention_matrix, loss_mask = exmp_batch

        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            x=image,
            patch_indices=image_coords,
            is_training=True,
        )["params"]
        param_count = sum(x.size for x in jax.tree_leaves(self.init_params))
        print(f"Number of parameters: {param_count}")

    def init_optimizer(self, freeze_backbone=False):
        # If we do not have any checkpoint to load, then create a new state
        if self.lr_schedule_type == "cosine":
            self.lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.lr,
                decay_steps=self.max_num_iterations,
                warmup_steps=self.warmup_steps,
                end_value=self.lr_end_value,
            )

        elif self.lr_schedule_type == "exponential":
            # decay_steps = 500k - 10k - 5k
            self.decay_steps = (
                self.max_num_iterations - self.cooldown_steps - self.warmup_steps
            )

            self.lr_schedule = warmup_exponential_decay_cooldown_scheduler(
                self.warmup_steps,
                self.lr,
                self.decay_steps,
                self.decay_rate,
                self.cooldown_steps,
                self.lr_end_value,
            )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),  # Clip gradients at norm 1
            optax.adamw(
                self.lr_schedule, b2=self.beta2, weight_decay=self.weight_decay
            ),
        )

        if freeze_backbone:
            print("Freezing backbone", flush=True)
            assert self.model_type == "classifier"

            partition_optimizers = {
                "trainable": self.optimizer,
                "frozen": optax.set_to_zero(),
            }

            param_partitions = traverse_util.path_aware_map(
                lambda path, _: "frozen"
                if "ClassificationHead_0" not in path
                else "trainable",
                self.init_params,
            )

            self.optimizer = optax.multi_transform(
                partition_optimizers, param_partitions
            )

        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params,
            tx=self.optimizer,
        )

    def loss_classifier(self, params, dropout_rng, batch: Batch, is_training: bool):
        logits = self.model.apply(
            {"params": params},
            x=batch.patches,
            patch_indices=batch.patch_coordinates,
            is_training=is_training,
            rngs=dropout_rng,
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch.labels
        ).mean()

        correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
        batch_size = batch.patches.shape[0]
        metrics = {
            "loss": ((loss * batch_size).astype(jnp.float32), batch_size),
            "accuracy": (correct_pred.sum().astype(jnp.float32), batch_size),
        }

        return loss, metrics

    def loss_autoregressor(self, params, dropout_rng, batch: Batch, is_training: bool):
        preds = self.model.apply(
            {"params": params},
            x=batch.patches,
            patch_indices=batch.patch_coordinates,
            is_training=is_training,
            attention_mask=batch.attention_masks,
            rngs=dropout_rng,
        )

        # We predict the next patch, so we need to slice the tensors
        loss = (
            preds[:, :-1, :] - batch.labels[:, 1:, :]
        ) ** 2  # shape = [bs, max_num_paches - 1, patch_size ** 2 * num_channels]

        # Apply mask on the loss so that gradients are computed
        # for the patches that are between the prefixed and padding patches
        loss = loss * batch.loss_masks[:, :-1, None]
        loss = jnp.mean(loss)

        batch_size = batch.patches.shape[0]
        metrics = {"loss": ((loss * batch_size).astype(jnp.float32), batch_size)}

        return loss, metrics

    def accumulate_gradients(self, loss_fn, state, batch, step_rng, num_minibatches):
        batch_size = batch.patches.shape[0]
        minibatch_size = batch_size // num_minibatches
        rngs = jax.random.split(step_rng, num_minibatches)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        grads = None
        metrics = None

        for minibatch_idx in range(num_minibatches):
            with jax.named_scope(f"minibatch_{minibatch_idx}"):
                # Split the batch into minibatches.
                start = minibatch_idx * minibatch_size
                end = start + minibatch_size
                minibatch = jax.tree_map(lambda x: x[start:end], batch)

                (_, minibatch_metrics), minibatch_grads = grad_fn(
                    state.params, rngs[minibatch_idx], minibatch
                )

                # Accumulate gradients and metrics across minibatches.
                if grads is None:
                    grads = minibatch_grads
                    metrics = minibatch_metrics
                else:
                    grads = jax.tree_map(jnp.add, grads, minibatch_grads)
                    metrics = jax.tree_map(jnp.add, metrics, minibatch_metrics)

        # TODO: this fails if num_minibatches does not evenly divide the batch_size
        grads = jax.tree_map(lambda g: g / num_minibatches, grads)
        return grads, metrics

    def train_step(self, loss_fn, state, step_rng, batch, num_minibatches, metrics):
        grads, step_metrics = self.accumulate_gradients(
            loss_fn, state, batch, step_rng, num_minibatches
        )

        new_state = state.apply_gradients(grads=grads)

        if metrics is None:
            metrics = step_metrics
        else:
            metrics = jax.tree_map(jnp.add, metrics, step_metrics)

        return new_state, metrics

    def eval_step(self, loss_fn, state, step_rng, batch):
        return loss_fn(state.params, step_rng, batch)[1]

    def train_model(self, train_loader, val_loader):
        if self.profile:
            jax.profiler.start_trace(self.tensorboard_path)

        # If we load the checkpoint, the first step will not be zero
        first_step = self.state.step

        # Pick the correct loss function
        # TODO somewhere else
        assert self.model_type in ["autoregressor", "classifier"]
        if self.model_type == "classifier":
            train_fn = partial(self.loss_classifier, is_training=True)
            eval_fn = partial(self.loss_classifier, is_training=False)
        else:
            train_fn = partial(self.loss_autoregressor, is_training=True)
            eval_fn = partial(self.loss_autoregressor, is_training=False)

        train_metrics = None

        for idx, batch in zip(
            trange(
                first_step,
                self.max_num_iterations,
                initial=first_step,
                total=self.max_num_iterations,
                desc="Training",
            ),
            train_loader,
        ):
            # Convert from tuple to Batch instance
            batch = Batch(*batch)

            # Normalise the patches
            if self.model_type == "autoregressor" and self.norm_pix_loss == "True":
                mean = jnp.mean(
                    batch.patches, axis=-1, keepdims=True
                )  # shape [bs, patches, 1]
                var = jnp.var(
                    batch.patches, axis=-1, keepdims=True
                )  # shape [bs, patches, 1]
                labels_norm = (batch.patches - mean) / (var + 1.0e-6) ** 0.5
                batch = batch.replace(labels=labels_norm)
            elif self.model_type == "autoregressor":
                batch = batch.replace(labels=batch.patches)

            if self.profile:
                context_manager = jax.profiler.StepTraceAnnotation(
                    "train_step",
                    step_num=idx + 1,
                )
            else:
                context_manager = nullcontext()
            with context_manager:
                # aux_output:
                self.rng, step_rng = random.split(self.rng)
                self.state, train_metrics = self.train_step(
                    train_fn,
                    self.state,
                    step_rng,
                    batch,
                    self.num_minibatches,
                    train_metrics,
                )

            # Logging
            if idx > first_step and (idx + 1) % self.log_every_n_steps == 0:
                for metric_name in train_metrics.keys():
                    metric_value = jax.device_get(train_metrics)[metric_name][0]
                    metric_count = jax.device_get(train_metrics)[metric_name][1]
                    self.logger.add_scalar(
                        f"{metric_name}/train", metric_value / metric_count, idx
                    )
                # Reinitialize the train_metrics
                train_metrics = jax.tree.map(lambda x: jnp.zeros_like(x), train_metrics)

            # Do evaluation once eval_steps
            if idx > first_step and (idx + 1) % self.eval_every_n_steps == 0:
                self.save_checkpoint(step=idx + 1)
                # Evaluate the model and return the eval metrics
                eval_metrics = self.eval_model(
                    eval_fn, self.state, prefetch(val_loader.as_numpy_iterator())
                )

                for metric_name in eval_metrics.keys():
                    metric_value = jax.device_get(eval_metrics)[metric_name][0]
                    metric_count = jax.device_get(eval_metrics)[metric_name][1]
                    self.logger.add_scalar(
                        f"{metric_name}/val", metric_value / metric_count, idx
                    )

                # Log the learning rate
                self.logger.add_scalar(
                    "Learning rate", float(self.lr_schedule(idx)), idx
                )

                # Flush the logger
                self.logger.flush()

        if self.profile:
            list(train_metrics.values())[0][0].block_until_ready()
            jax.profiler.stop_trace()

        self.checkpointer.wait_until_finished()

        self.logger.close()

    def eval_model(self, loss_fn, state, data_loader):
        # Test model on all images of a data loader and return avg mse or acc
        batches_to_skip = np.random.randint(1, 50)
        metrics = None

        for idx, batch in enumerate(tqdm(data_loader, desc="Validation", leave=False)):
            # Convert from tuple to Batch instance
            batch = Batch(*batch)

            # Normalise the patches
            if self.model_type == "autoregressor" and self.norm_pix_loss == "True":
                mean = jnp.mean(
                    batch.patches, axis=-1, keepdims=True
                )  # shape [bs, patches, 1]
                var = jnp.var(
                    batch.patches, axis=-1, keepdims=True
                )  # shape [bs, patches, 1]
                labels_norm = (batch.patches - mean) / (var + 1.0e-6) ** 0.5
                batch = batch.replace(labels=labels_norm)
            elif self.model_type == "autoregressor":
                batch = batch.replace(labels=batch.patches)

            if self.model_type == "autoregressor":
                # Always plot the same few images for easy comparison across training stages
                if idx == 0:
                    self.visualize_pretraining_output(
                        batch.patches,
                        batch.patch_coordinates,
                        batch.attention_masks,
                        plot_random_images=False,
                    )

                # Also plot a few random images for a more representative sample
                if idx == batches_to_skip:
                    self.visualize_pretraining_output(
                        batch.patches,
                        batch.patch_coordinates,
                        batch.attention_masks,
                        plot_random_images=True,
                    )
            _, batch_metrics = loss_fn(
                params=state.params, dropout_rng=None, batch=batch
            )
            if metrics is None:
                metrics = batch_metrics
            else:
                metrics = jax.tree_map(jnp.add, metrics, batch_metrics)

        return metrics

    def save_checkpoint(self, step):
        self.checkpointer.save(self.checkpoints_path / f"step_{step}", self.state)

    def load_checkpoint(
        self,
        checkpoint_path_to_load,
        model_type,
        load_only_params=False,
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

            restored["params"] = translate(self.state.params, restored["params"])

            self.state = self.state.replace(
                params=restored["params"],
            )
        else:
            print("Loading full train state")
            restored["params"] = translate(self.state.params, restored["params"])

            if model_type == "autoregressor":
                restored["opt_state"][1][0]["mu"] = translate(
                    self.state.opt_state[1][0].mu,
                    restored["opt_state"][1][0]["mu"],
                )

                restored["opt_state"][1][0]["nu"] = translate(
                    self.state.opt_state[1][0].nu,
                    restored["opt_state"][1][0]["nu"],
                )

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
            is_training=False,
            attention_mask=attention_matrices,
            rngs=None,
        )
        # Prepend a black patch to the predictions to make the visualization line up
        preds = jnp.concatenate(
            (jnp.zeros((preds.shape[0], 1, preds.shape[2])), preds), axis=1
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

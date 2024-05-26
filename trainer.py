from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import traverse_util
from flax.training import train_state
from jax import random
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

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

        # Get empty model based on model_type
        if model_type == "autoregressor":
            self.model = PretrainingModel(**model_hparams)
        elif model_type == "no_transformer_classifier":
            print("Look mom, no transformer!")
            self.model = NoTransformerClassificationModel(**model_hparams)
        else:
            self.model = ClassificationModel(**model_hparams)

        # Initialize model, logger and optimizer
        self.init_model(dummy_batch, model_type)
        self.init_logger()
        self.init_optimizer()
        if checkpoint_path_to_load:
            self.load_checkpoint(
                checkpoint_path_to_load, model_type, load_only_params, freeze_backbone
            )

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
            end_value=self.lr_end_value,
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),  # Clip gradients at norm 1
            optax.adamw(
                self.lr_schedule, b2=self.beta2, weight_decay=self.weight_decay
            ),
        )

        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params,
            tx=self.optimizer,
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
        if "classifier" in self.model_type:
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

        metric_to_eval = "mse" if "mse" in self.metrics_keys else "acc"

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

            restored["params"] = translate(self.state.params, restored["params"])

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
            restored["params"] = translate(self.state.params, restored["params"])

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
            training=False,
            mask=attention_matrices,
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

            plt.imshow(tgt_norm.transpose(1, 2, 0))
            plt.savefig(f"target_{i}.png")
            plt.imshow(pred_norm.transpose(1, 2, 0))
            plt.savefig(f"pred_{i}.png")

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

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


class TrainerAutoregressor:
    def __init__(
        self, dummy_imgs, lr=1e-3, seed=42, norm_pix_loss=False, **model_hparams
    ):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        self.norm_pix_loss = norm_pix_loss

        self.log_dir = os.path.join(str(Path.cwd()), "checkpoints/autoregressor/")

        # Empty model
        self.model = PretrainingModel(**model_hparams)

        # Initialize model and logger
        self.init_model(dummy_imgs)
        self.init_logger()

        # Jitting train and eval steps
        self.train_step = jax.jit(self.train_step)
        self.eval_step = jax.jit(self.eval_step)

    def init_logger(self):
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.logger.add_text("learning rate", f"{self.lr}")
        self.logger.add_text("seed", f"{self.seed}")

    def init_model(self, exmp_imgs):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng}, exmp_imgs, training=True
        )["params"]
        self.state = None

    def get_loss(self, params, rng, batch, train):
        """
        imgs, targets, mask: [bs, 28, 28]
        """
        imgs, targets, mask = batch

        # Normalize the target
        if self.norm_pix_loss:
            mean = jnp.mean(targets, axis=(-2, -1), keepdims=True)  # shape [bs, 1, 1]
            var = jnp.var(targets, axis=(-2, -1), keepdims=True)  # shape [bs, 1, 1]
            targets = (targets - mean) / (var + 1.0e-6) ** 0.5

        rng, dropout_apply_rng = random.split(rng)

        preds = self.model.apply(
            {"params": params},
            imgs,
            training=train,
            mask=mask,
            rngs={"dropout": dropout_apply_rng},
        )

        # Pixel-wise MSE
        loss = (preds - targets) ** 2  # shape = [bs, 28, 28]
        loss = jnp.mean(loss)

        return loss, rng

    def train_step(self, state, rng, batch):
        def loss_fn(params):
            return self.get_loss(params, rng, batch, train=True)

        # Get output of loss function and gradients of the loss
        (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        # Update parameters and batch statistics
        state = state.apply_gradients(grads=grads)
        return state, rng, loss

    def eval_step(self, state, rng, batch):
        # Return the mse for a single batch
        mse, rng = self.get_loss(state.params, rng, batch, train=False)
        return mse, rng

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

    def train_model(self, train_loader, val_loader, num_epochs=200):
        # Train model for defined number of epochs
        self.init_optimizer()
        # Track best eval accuracy
        best_eval = 999999999
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            train_metrics = self.train_epoch(train_loader)
            eval_mse = self.eval_model(val_loader)
            if eval_mse <= best_eval:
                best_eval = eval_mse
                self.save_model(step=epoch_idx)

            # Log the loss
            self.logger.add_scalar("MSE/train", train_metrics["mse"], epoch_idx)
            self.logger.add_scalar("MSE/val", eval_mse, epoch_idx)

    def train_epoch(self, train_loader):
        # Train model for one epoch, and print avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.state, self.rng, loss = self.train_step(self.state, self.rng, batch)
            metrics["mse"].append(loss)

        metrics = jax.device_get(metrics)
        metrics = {key: np.array(metric).mean() for key, metric in metrics.items()}

        return metrics

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg accuracy
        total_mse, count = 0, 0
        for batch in data_loader:
            mse, self.rng = self.eval_step(self.state, self.rng, batch)
            total_mse += mse * batch[0].shape[0]
            count += batch[0].shape[0]

        res = (total_mse / count).item()
        return res

    def save_model(self, step=0):
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


class TrainerClassifier:
    def __init__(self, dummy_imgs, lr=1e-3, seed=42, **model_hparams):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)

        self.log_dir = os.path.join(str(Path.cwd()), "checkpoints/classification/")

        # Empty model
        self.model = ClassificationModel(**model_hparams)

        # Initialize model and logger
        self.init_model(dummy_imgs)
        self.init_logger()

        # Jitting train and eval steps
        self.train_step = jax.jit(self.train_step)
        self.eval_step = jax.jit(self.eval_step)

    def init_logger(self):
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.logger.add_text("learning rate", f"{self.lr}")
        self.logger.add_text("seed", f"{self.seed}")

    def init_model(self, exmp_imgs):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng}, exmp_imgs, training=True
        )["params"]
        self.state = None

    def get_loss(self, params, rng, batch, train):
        imgs, labels = batch
        rng, dropout_apply_rng = random.split(rng)
        logits = self.model.apply(
            {"params": params},
            imgs,
            training=train,
            rngs={"dropout": dropout_apply_rng},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels).mean()

        return loss, (acc, rng)

    def train_step(self, state, rng, batch):
        def loss_fn(params):
            return self.get_loss(params, rng, batch, train=True)

        # Get output of loss function and gradients of the loss
        (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        # Update parameters and batch statistics
        state = state.apply_gradients(grads=grads)
        return state, rng, loss, acc

    def eval_step(self, state, rng, batch):
        # Return the accuracy for a single batch
        _, (acc, rng) = self.get_loss(state.params, rng, batch, train=False)
        return rng, acc

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

    def train_model(self, train_loader, val_loader, num_epochs=200):
        # Train model for defined number of epochs
        self.init_optimizer()
        # Track best eval accuracy
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            train_metrics = self.train_epoch(train_loader, epoch=epoch_idx)
            eval_acc = self.eval_model(val_loader)
            if eval_acc >= best_eval:
                best_eval = eval_acc
                self.save_model(step=epoch_idx)

            # Log the loss
            self.logger.add_scalar("Acc/train", train_metrics["acc"], epoch_idx)
            self.logger.add_scalar("Acc/val", eval_acc, epoch_idx)

        self.logger.flush()

    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and print avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.state, self.rng, loss, acc = self.train_step(
                self.state, self.rng, batch
            )
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)

        metrics = jax.device_get(metrics)
        metrics = {key: np.array(metric).mean() for key, metric in metrics.items()}

        return metrics

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg accuracy
        correct_class, count = 0, 0
        for batch in data_loader:
            self.rng, acc = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
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

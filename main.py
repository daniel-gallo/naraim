import argparse

import jax.numpy as jnp

from dataset import get_dataloader
from trainer import Trainer


def train_autoregressor(
    batch_size: int,
    num_epochs: int,
    model_type: str,
    lr: float,
    seed: int,
    log_every_n_steps: int,
    norm_pix_loss: bool,
    embedding_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    patch_size: int,
    max_num_patches: int,
    num_channels: int,
    dataset_name: str,
):
    train_dataloader = get_dataloader(
        dataset_name,
        pretraining=True,
        train=True,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    val_dataloader = get_dataloader(
        dataset_name,
        pretraining=True,
        train=False,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    trainer = Trainer(
        dummy_batch=next(iter(train_dataloader)),
        model_type=model_type,
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        norm_pix_loss=norm_pix_loss,
        dtype=jnp.float32,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        num_channels=num_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_rate,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=num_epochs)
    val_mse = trainer.eval_model(val_dataloader)
    print(f"Final MSE: {val_mse}")


def train_classifier(
    batch_size: int,
    num_epochs: int,
    model_type: str,
    lr: float,
    seed: int,
    log_every_n_steps: int,
    norm_pix_loss: bool,
    embedding_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    num_heads: int,
    num_categories: int,
    dropout_rate: float,
    patch_size: int,
    max_num_patches: int,
    dataset_name: str,
):
    train_dataloader = get_dataloader(
        dataset_name,
        pretraining=False,
        train=True,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    val_dataloader = get_dataloader(
        dataset_name,
        pretraining=False,
        train=False,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    trainer = Trainer(
        dummy_batch=next(iter(train_dataloader)),
        model_type=model_type,
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        norm_pix_loss=norm_pix_loss,
        dtype=jnp.float32,
        max_num_patches=max_num_patches,
        num_categories=num_categories,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_rate,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=num_epochs)
    val_acc = trainer.eval_model(val_dataloader)
    print(f"Final accuracy: {val_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")

    parser.add_argument(
        "--model_type",
        type=str,
        default="autoregressor",
        choices=["classifier", "autoregressor"],
        help="What to train",
    )

    parser.add_argument(
        "--num_categories",
        type=int,
        default=10,
        help="Number of classification categories",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=768,
        help="Embedding dimension of the tokens",
    )
    parser.add_argument(
        "--hidden_dimension",
        type=int,
        default=128,
        help="Hidden dimension of the model",
    )
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument(
        "--max_num_patches", type=int, default=256, help="Max number of patches"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--norm_pix_loss",
        type=bool,
        default=True,
        help="Whether to use a normalized-pixel loss",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps until next logging",
    )

    parser.add_argument("--num_channels", type=int, default=1, help="Num channels")

    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["fashion_mnist", "imagenet"],
        help="Dataset name",
    )

    args = parser.parse_args()

    if args.model_type == "autoregressor":
        train_autoregressor(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            model_type=args.model_type,
            lr=args.lr,
            seed=args.seed,
            log_every_n_steps=args.log_every_n_steps,
            norm_pix_loss=args.norm_pix_loss,
            embedding_dimension=args.embedding_dimension,
            hidden_dimension=args.hidden_dimension,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            patch_size=args.patch_size,
            max_num_patches=args.max_num_patches,
            num_channels=args.num_channels,
            dataset_name=args.dataset,
        )

    elif args.model_type == "classifier":
        train_classifier(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            model_type=args.model_type,
            lr=args.lr,
            seed=args.seed,
            log_every_n_steps=args.log_every_n_steps,
            norm_pix_loss=args.norm_pix_loss,
            embedding_dimension=args.embedding_dimension,
            hidden_dimension=args.hidden_dimension,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_categories=args.num_categories,
            dropout_rate=args.dropout_rate,
            patch_size=args.patch_size,
            max_num_patches=args.max_num_patches,
            dataset_name=args.dataset,
        )


# class TrainerClassifier:
#     def __init__(self, dummy_batch, lr, seed, log_every_n_steps, **model_hparams):
#         super().__init__()
#         self.lr = lr
#         self.seed = seed
#         self.rng = jax.random.PRNGKey(self.seed)
#         self.log_every_n_steps = log_every_n_steps

#         self.log_dir = os.path.join(str(Path.cwd()), "checkpoints/classification/")

#         # Empty model
#         self.model = ClassificationModel(**model_hparams)

#         # Initialize model and logger
#         self.init_model(dummy_batch)
#         self.init_logger()

#         # Jitting train and eval steps
#         self.train_step = jax.jit(self.train_step)
#         self.eval_step = jax.jit(self.eval_step)

#     def init_logger(self):
#         self.logger = SummaryWriter(log_dir=self.log_dir)

#     def init_model(self, exmp_batch):
#         self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
#         patches, patch_indices, _ = exmp_batch

#         self.init_params = self.model.init(
#             {"params": init_rng, "dropout": dropout_init_rng},
#             patches,
#             patch_indices,
#             training=True,
#         )["params"]
#         self.state = None

#     def get_loss(self, params, rng, batch, train):
#         imgs, patch_indices, labels = batch
#         rng, dropout_apply_rng = random.split(rng)
#         logits = self.model.apply(
#             {"params": params},
#             imgs,
#             patch_indices,
#             training=train,
#             rngs={"dropout": dropout_apply_rng},
#         )

#         loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
#         acc = (logits.argmax(axis=-1) == labels).mean()

#         return loss, (acc, rng)

#     def train_step(self, state, rng, batch):
#         def loss_fn(params):
#             return self.get_loss(params, rng, batch, train=True)

#         # Get output of loss function and gradients of the loss
#         (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
#             state.params
#         )
#         # Update parameters and batch statistics
#         state = state.apply_gradients(grads=grads)
#         return state, rng, loss, acc

#     def eval_step(self, state, rng, batch):
#         # Return the accuracy for a single batch
#         _, (acc, _) = self.get_loss(state.params, rng, batch, train=False)
#         return acc

#     def init_optimizer(self):
#         # TODO: lr_scheduler?

#         optimizer = optax.chain(
#             optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
#             optax.adamw(self.lr),
#         )
#         # Initialize training state
#         self.state = train_state.TrainState.create(
#             apply_fn=self.model.apply,
#             params=self.init_params if self.state is None else self.state.params,
#             tx=optimizer,
#         )

#     def train_model(self, train_loader, val_loader, num_epochs=200):
#         # Train model for defined number of epochs
#         self.init_optimizer()
#         # Track best eval accuracy
#         best_eval = 0.0
#         hparams_dict = {"learning_rate": self.lr, "seed": self.seed}
#         best_metrics = {"Max_Acc/train": 0, "Max_Acc/val": 0}

#         for epoch_idx in tqdm(range(1, num_epochs + 1)):
#             train_metrics = self.train_epoch(train_loader, epoch=epoch_idx)
#             eval_acc = self.eval_model(val_loader)
#             if eval_acc >= best_eval:
#                 best_eval = eval_acc
#                 self.save_model(step=epoch_idx)
#                 best_metrics["Max_Acc/train"] = train_metrics["acc"]
#                 best_metrics["Max_Acc/val"] = eval_acc

#             # Log the loss
#             self.logger.add_scalar("Acc/val", eval_acc, epoch_idx)

#         self.logger.add_hparams(hparams_dict, best_metrics)
#         self.logger.flush()
#         self.logger.close()

#     def train_epoch(self, train_loader, epoch):
#         # Train model for one epoch, and print avg loss and accuracy
#         metrics = defaultdict(list)
#         for idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
#             self.state, self.rng, loss, acc = self.train_step(
#                 self.state, self.rng, batch
#             )
#             metrics["loss"].append(loss)
#             metrics["acc"].append(acc)

#             if idx > 1 and idx % self.log_every_n_steps == 0:
#                 step = (epoch - 1) * (len(train_loader) // 10) + idx // 10
#                 self.logger.add_scalar(
#                     "Loss/train", np.array(jax.device_get(metrics)["loss"]).mean(), step
#                 )
#                 self.logger.add_scalar(
#                     "Acc/train", np.array(jax.device_get(metrics)["acc"]).mean(), step
#                 )

#         metrics = jax.device_get(metrics)
#         metrics = {key: np.array(metric).mean() for key, metric in metrics.items()}

#         return metrics

#     def eval_model(self, data_loader):
#         # Test model on all images of a data loader and return avg accuracy
#         correct_class, count = 0, 0
#         eval_rng = jax.random.PRNGKey(self.seed)  # same rng for evaluation

#         for batch in data_loader:
#             acc = self.eval_step(self.state, eval_rng, batch)
#             correct_class += acc * batch[0].shape[0]
#             count += batch[0].shape[0]
#         eval_acc = (correct_class / count).item()
#         return eval_acc

#     def save_model(self, step=0):
#         # Save current model at certain training iteration
#         checkpoints.save_checkpoint(
#             ckpt_dir=self.log_dir, target=self.state.params, step=step, overwrite=True
#         )

#     def load_model(self):
#         # Load model
#         params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)

#         self.state = train_state.TrainState.create(
#             apply_fn=self.model.apply,
#             params=params,
#             tx=self.state.tx
#             if self.state
#             else optax.adamw(self.lr),  # Default optimizer
#         )

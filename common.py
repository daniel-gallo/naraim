import jax.numpy as jnp
import optax


def warmup_exponential_decay_cooldown_scheduler(
    warmup_steps, peak_lr, decay_steps, decay_rate, cooldown_steps, min_lr
):
    """
    Function that creates a learning rate schedule which includes warmup:
    - warmup (linear scheduler);
    - exponential decay;
    - cooldown (linear scheduler).

    Parameters:
    - warmup_steps: Number of steps for the warmup phase
    - peak_lr: Base learning rate after warmup.
    - decay_steps: Number of steps for the exponential decay phase.
    - decay_rate: Rate of exponential decay.
    - cooldown_steps: Number of steps for the cooldown phase.
    - min_lr: Minimum learning rate during cooldown.

    Returns:
    - schedule_fn: A function that takes the current step and returns the learning rate.
    """
    warmup_schedule = optax.linear_schedule(
        init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps
    )

    def exponential_decay_schedule(step):
        return peak_lr * (decay_rate ** (step / decay_steps))

    def cooldown_schedule(step):
        cooldown_lr = jnp.maximum(
            min_lr,
            peak_lr
            * (decay_rate ** (decay_steps / decay_steps))
            * (1 - (step / cooldown_steps)),
        )
        return cooldown_lr

    def schedule_fn(step):
        warmup_lr = warmup_schedule(step)
        decay_step = step - warmup_steps
        decay_lr = jnp.where(
            decay_step >= 0, exponential_decay_schedule(decay_step), warmup_lr
        )
        cooldown_step = decay_step - decay_steps
        lr = jnp.where(cooldown_step >= 0, cooldown_schedule(cooldown_step), decay_lr)
        return lr

    return schedule_fn

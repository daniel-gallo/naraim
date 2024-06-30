import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    ### Testing the lr_scheduler
    num_iters = 50_000
    warmup_steps = 500
    cooldown_steps = 1_000

    decay_steps = num_iters - warmup_steps - cooldown_steps

    base_lr = 1e-3
    clip_value = 1.0
    decay_rate = 0.1

    lr_scheduler = warmup_exponential_decay_cooldown_scheduler(
        warmup_steps, base_lr, decay_steps, decay_rate, cooldown_steps, min_lr=0.0
    )

    ### Small training loop
    params = jax.random.normal(jax.random.PRNGKey(0), (3, 3))

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_value),
        optax.adamw(lr_scheduler, b2=0.98, weight_decay=0.01),
    )

    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(lambda params: jnp.sum(params**2))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    num_steps = warmup_steps + decay_steps + cooldown_steps

    steps = []
    lrs = []
    for step in range(num_steps + 1):
        batch = jnp.ones((3, 3))  # Dummy batch data
        params, opt_state, loss = update(params, opt_state)
        current_lr = lr_scheduler(step)
        if step % 1000 == 0:
            print(f"Step {step} | Loss: {loss} | Learning Rate: {current_lr}")
            steps.append(step)
            lrs.append(current_lr)

    ### Plot the LR
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, label="Learning rate")

    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Warmup, Exponential Decay, and Cooldown Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_exponential.png")
    plt.show()

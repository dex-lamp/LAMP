"""
Train Hand Action VAE with Flax + Optax.

Usage:
    python vae/scripts/train_jax.py \
        --train_dir data/example_task/demos/success/train \
        --test_dir data/example_task/demos/success/test \
        --output_dir outputs/hand_vae_jax
"""

import argparse
import importlib.util
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

_vae_root = os.path.join(os.path.dirname(__file__), "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionVAE = _load("model/hand_vae_jax.py", "hand_vae_jax").HandActionVAE
count_params = _load("model/hand_vae_jax.py", "hand_vae_jax").count_params
_data_io = _load("model/data_io.py", "data_io")
load_window_arrays = _data_io.load_window_arrays
iterate_minibatches = _data_io.iterate_minibatches
_compat = _load("model/checkpoint_compat.py", "checkpoint_compat")
save_jax_checkpoint_payload = _compat.save_jax_checkpoint
_utils = _load("model/utils.py", "utils")
cosine_scheduler = _utils.cosine_scheduler
beta_annealing_schedule = _utils.beta_annealing_schedule


class TrainState(train_state.TrainState):
    """Light wrapper for the VAE train state."""


def get_args():
    p = argparse.ArgumentParser(description="Train Hand Action VAE with JAX")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/hand_vae_jax")
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--beta_warmup", type=int, default=2000)
    p.add_argument("--encoder_type", type=str, default="mlp", choices=["mlp", "causal_conv"])
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--recon_aux_weight", type=float, default=0.0)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    return p.parse_args()


def build_model_args(args):
    return {
        "action_dim": args.action_dim,
        "window_size": args.window_size,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "encoder_type": args.encoder_type,
        "num_hidden_layers": args.num_hidden_layers,
        "recon_aux_weight": args.recon_aux_weight,
        "free_bits": args.free_bits,
    }


def save_training_curves(history, output_dir, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curves")
        return

    steps = history["steps"]
    eval_steps = history["eval_steps"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"JAX VAE Training Curves  (noise_std={args.noise_std}, beta={args.beta}, "
        f"latent_dim={args.latent_dim}, encoder={args.encoder_type})",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(steps, history["train_total"], alpha=0.3, color="blue", linewidth=0.5)
    window = min(100, len(steps) // 10)
    if window > 1:
        from numpy import convolve, ones

        kernel = ones(window) / window
        smoothed = convolve(history["train_total"], kernel, mode="valid")
        ax.plot(steps[: len(smoothed)], smoothed, color="blue", linewidth=1.5, label="Smoothed")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Total Loss (Reconstruction + beta x KL)")
    ax.grid(True, alpha=0.3)
    if window > 1:
        ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, history["train_recon"], alpha=0.3, color="green", linewidth=0.5)
    if window > 1:
        smoothed = convolve(history["train_recon"], kernel, mode="valid")
        ax.plot(steps[: len(smoothed)], smoothed, color="green", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_recon"], "ro-", markersize=4, linewidth=1.5, label="Validation")
        ax.axhline(y=history["val_copy"][0], color="gray", linestyle="--", linewidth=1, label="Copy Baseline")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.set_title("Reconstruction Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(steps, history["train_kl"], alpha=0.3, color="orange", linewidth=0.5)
    if window > 1:
        smoothed = convolve(history["train_kl"], kernel, mode="valid")
        ax.plot(steps[: len(smoothed)], smoothed, color="orange", linewidth=1.5, label="Train (smoothed)")
    if eval_steps:
        ax.plot(eval_steps, history["val_kl"], "ro-", markersize=4, linewidth=1.5, label="Validation")
    ax.set_ylabel("KL Divergence")
    ax.set_xlabel("Step")
    ax.set_title("KL(q(z|x) || N(0,1))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, history["train_beta"], color="purple", linewidth=1.5)
    ax.set_ylabel("beta")
    ax.set_xlabel("Step")
    ax.set_title("KL Weight Schedule")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("Cosine LR Schedule")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    beta_kl = [b * k for b, k in zip(history["train_beta"], history["train_kl"])]
    if window > 1:
        smoothed_recon = convolve(history["train_recon"], kernel, mode="valid")
        smoothed_bkl = convolve(beta_kl, kernel, mode="valid")
        ax.plot(steps[: len(smoothed_recon)], smoothed_recon, color="green", linewidth=1.5, label="Recon")
        ax.plot(steps[: len(smoothed_bkl)], smoothed_bkl, color="red", linewidth=1.5, label="beta x KL")
    else:
        ax.plot(steps, history["train_recon"], color="green", linewidth=1, label="Recon")
        ax.plot(steps, beta_kl, color="red", linewidth=1, label="beta x KL")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.set_title("Loss Decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves_jax.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {fig_path}")


def save_summary(history, output_dir, args, final_metrics):
    summary = {
        "backend": "jax",
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "output_dir": args.output_dir,
        "model_args": build_model_args(args),
        "train_args": vars(args),
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def create_train_state(args, lr_schedule):
    model = HandActionVAE(**build_model_args(args))
    init_window = jnp.zeros((1, args.window_size, args.action_dim), dtype=jnp.float32)
    init_target = jnp.zeros((1, args.action_dim), dtype=jnp.float32)
    init_rng = jax.random.PRNGKey(args.seed)
    variables = model.init({"params": init_rng, "sample": init_rng}, init_window, init_target, beta=args.beta)
    params = variables["params"]
    lr_schedule_jax = jnp.asarray(lr_schedule, dtype=jnp.float32)

    def schedule_fn(step):
        step = jnp.minimum(step, lr_schedule_jax.shape[0] - 1)
        return lr_schedule_jax[step]

    tx = optax.chain(
        optax.clip_by_global_norm(args.clip_grad),
        optax.adamw(learning_rate=schedule_fn, weight_decay=args.weight_decay),
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def make_train_step(model, noise_std):
    @jax.jit
    def train_step(state, window, target, beta, step_key):
        noise_key, sample_key = jax.random.split(step_key)
        if noise_std > 0:
            window = window + jax.random.normal(noise_key, window.shape, dtype=window.dtype) * noise_std

        def loss_fn(params):
            _, recon_loss, kl_loss, total_loss, _, _ = model.apply(
                {"params": params},
                window,
                target,
                beta=beta,
                rngs={"sample": sample_key},
            )
            return total_loss, (recon_loss, kl_loss)

        (total_loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }
        return state, metrics

    return train_step


def evaluate_model(model, params, windows, targets, batch_size, beta, seed):
    total_recon = 0.0
    total_kl = 0.0
    total_copy = 0.0
    total_n = 0
    eval_rng = jax.random.PRNGKey(seed)
    np_rng = np.random.RandomState(seed)

    for window_np, target_np in iterate_minibatches(
        windows,
        targets,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        rng=np_rng,
    ):
        eval_rng, sample_key = jax.random.split(eval_rng)
        window = jnp.asarray(window_np)
        target = jnp.asarray(target_np)
        _, recon_loss, kl_loss, _, _, _ = model.apply(
            {"params": params},
            window,
            target,
            beta=beta,
            rngs={"sample": sample_key},
        )

        copy_mse = float(np.mean((window_np[:, -1, :] - target_np) ** 2))
        batch_size_actual = window_np.shape[0]
        total_recon += float(recon_loss) * batch_size_actual
        total_kl += float(kl_loss) * batch_size_actual
        total_copy += copy_mse * batch_size_actual
        total_n += batch_size_actual

    return total_recon / total_n, total_kl / total_n, total_copy / total_n


def save_jax_checkpoint(state, rng, step, output_dir, args):
    ckpt_dir = os.path.join(output_dir, "jax_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": int(step),
        "rng": np.asarray(rng),
        "model_args": build_model_args(args),
        "train_args": vars(args),
        "backend": "jax",
    }
    return save_jax_checkpoint_payload(ckpt_dir, payload, step)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")

    train_windows, train_targets = load_window_arrays(args.train_dir, window_size=args.window_size)
    test_windows, test_targets = load_window_arrays(args.test_dir, window_size=args.window_size)

    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)
    beta_schedule = beta_annealing_schedule(args.beta, args.total_steps, warmup_steps=args.beta_warmup)
    model, state = create_train_state(args, lr_schedule)
    train_step = make_train_step(model, args.noise_std)

    print(f"Model parameters: {count_params(state.params):,}")
    val_recon, val_kl, val_copy = evaluate_model(
        model,
        state.params,
        test_windows,
        test_targets,
        batch_size=args.batch_size,
        beta=args.beta,
        seed=args.seed,
    )
    print(f"Copy-baseline MSE: {val_copy:.6f}  (this is the bar to beat)\n")

    print(f"Training for {args.total_steps} steps")
    print(f"  Train: {len(train_windows)} samples, Test: {len(test_windows)} samples")
    print(f"  Latent dim: {args.latent_dim}, Beta: 0->{args.beta} over {args.beta_warmup} steps")
    print(f"  Noise std: {args.noise_std}\n")

    history = {
        "steps": [],
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "train_beta": [],
        "train_lr": [],
        "eval_steps": [],
        "val_recon": [],
        "val_kl": [],
        "val_copy": [],
    }

    np_rng = np.random.RandomState(args.seed)
    train_iter = iter(
        iterate_minibatches(
            train_windows,
            train_targets,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            rng=np_rng,
        )
    )
    step_rng = jax.random.PRNGKey(args.seed + 1)

    for step in range(args.total_steps):
        try:
            window_np, target_np = next(train_iter)
        except StopIteration:
            train_iter = iter(
                iterate_minibatches(
                    train_windows,
                    train_targets,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    rng=np_rng,
                )
            )
            window_np, target_np = next(train_iter)

        beta_value = float(beta_schedule[step])
        lr_value = float(lr_schedule[step])
        window = jnp.asarray(window_np)
        target = jnp.asarray(target_np)
        step_rng, step_key = jax.random.split(step_rng)
        state, metrics = train_step(state, window, target, beta_value, step_key)

        history["steps"].append(step)
        history["train_total"].append(float(metrics["total"]))
        history["train_recon"].append(float(metrics["recon"]))
        history["train_kl"].append(float(metrics["kl"]))
        history["train_beta"].append(beta_value)
        history["train_lr"].append(lr_value)

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] total={float(metrics['total']):.6f}  "
                f"recon={float(metrics['recon']):.6f}  kl={float(metrics['kl']):.4f}  "
                f"beta={beta_value:.4f}  lr={lr_value:.2e}"
            )

        if step % args.eval_freq == 0 and step > 0:
            vr, vk, vc = evaluate_model(
                model,
                state.params,
                test_windows,
                test_targets,
                batch_size=args.batch_size,
                beta=beta_value,
                seed=args.seed + step,
            )
            history["eval_steps"].append(step)
            history["val_recon"].append(vr)
            history["val_kl"].append(vk)
            history["val_copy"].append(vc)
            print(f"           val_recon={vr:.6f}  val_kl={vk:.4f}  copy_baseline={vc:.6f}")

        if step > 0 and step % args.save_freq == 0:
            saved_path = save_jax_checkpoint(state, step_rng, step, args.output_dir, args)
            print(f"           checkpoint saved -> {saved_path}")

    save_jax_checkpoint(state, step_rng, args.total_steps, args.output_dir, args)
    final_beta = float(beta_schedule[min(args.total_steps - 1, len(beta_schedule) - 1)])
    vr, vk, vc = evaluate_model(
        model,
        state.params,
        test_windows,
        test_targets,
        batch_size=args.batch_size,
        beta=final_beta,
        seed=args.seed + args.total_steps,
    )
    history["eval_steps"].append(args.total_steps)
    history["val_recon"].append(vr)
    history["val_kl"].append(vk)
    history["val_copy"].append(vc)

    final_metrics = {
        "val_recon": vr,
        "val_kl": vk,
        "copy_baseline": vc,
        "beats_copy_baseline": vr < vc,
        "num_train_samples": int(len(train_windows)),
        "num_test_samples": int(len(test_windows)),
    }

    print(f"\nFinal: val_recon={vr:.6f}  val_kl={vk:.4f}  copy_baseline={vc:.6f}")
    print(f"  VAE {'beats' if vr < vc else 'does NOT beat'} copy baseline ({vr:.6f} vs {vc:.6f})")

    final_ckpt = os.path.join(args.output_dir, "jax_checkpoints", "checkpoint.pkl")
    save_training_curves(history, args.output_dir, args)
    save_summary(history, args.output_dir, args, final_metrics)
    print(f"\nDone! JAX checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()

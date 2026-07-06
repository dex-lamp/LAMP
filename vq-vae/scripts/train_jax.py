"""Train the hand VQ-VAE with Flax + Optax."""

import argparse
import importlib.util
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state


_VQVAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_VQVAE_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model_mod = _load("model/hand_vqvae.py", "hand_vqvae_jax")
_data_mod = _load("model/hand_dataset.py", "hand_dataset_vqvae_jax")
_utils_mod = _load("model/utils.py", "vqvae_utils_jax")
_ckpt_mod = _load("model/checkpoint_compat.py", "vqvae_checkpoint_compat")

HandVQVAE = _model_mod.HandVQVAE
count_params = _model_mod.count_params
load_hand_action_array = _data_mod.load_hand_action_array
iterate_minibatches = _data_mod.iterate_minibatches
cosine_scheduler = _utils_mod.cosine_scheduler
save_jax_checkpoint = _ckpt_mod.save_jax_checkpoint


class TrainState(train_state.TrainState):
    vq_state: flax.core.FrozenDict


def get_args():
    p = argparse.ArgumentParser(description="Train hand VQ-VAE with JAX")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/hand_vqvae_jax")
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_vq_layers", type=int, default=2)
    p.add_argument("--codebook_size", type=int, default=4)
    p.add_argument("--vae_layer_num", type=int, default=5)
    p.add_argument("--encoder_loss_multiplier", type=float, default=1.0)
    p.add_argument("--recon_loss_weight", type=float, default=3.0)
    p.add_argument("--vq_loss_weight", type=float, default=5.0)
    p.add_argument("--ema_decay", type=float, default=0.8)
    p.add_argument("--threshold_ema_dead_code", type=float, default=0.0)
    p.add_argument("--normalize_actions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--total_steps", type=int, default=1500)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=233)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--save_freq", type=int, default=2000)
    return p.parse_args()


def build_model_args(args):
    return {
        "action_dim": args.action_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "num_vq_layers": args.num_vq_layers,
        "codebook_size": args.codebook_size,
        "vae_layer_num": args.vae_layer_num,
        "encoder_loss_multiplier": args.encoder_loss_multiplier,
        "recon_loss_weight": args.recon_loss_weight,
        "vq_loss_weight": args.vq_loss_weight,
        "normalize_actions": args.normalize_actions,
        "ema_decay": args.ema_decay,
        "threshold_ema_dead_code": args.threshold_ema_dead_code,
    }


def create_train_state(args, lr_schedule):
    model = HandVQVAE(**build_model_args(args))
    init_x = jnp.zeros((1, args.action_dim), dtype=jnp.float32)
    variables = model.init({"params": jax.random.PRNGKey(args.seed)}, init_x, training=False)
    params = variables["params"]
    vq_state = variables["vq_state"]
    lr_schedule_jax = jnp.asarray(lr_schedule, dtype=jnp.float32)

    def schedule_fn(step):
        step = jnp.minimum(step, lr_schedule_jax.shape[0] - 1)
        return lr_schedule_jax[step]

    tx_parts = []
    if args.clip_grad > 0:
        tx_parts.append(optax.clip_by_global_norm(args.clip_grad))
    tx_parts.append(optax.adamw(learning_rate=schedule_fn, b1=0.95, b2=0.999, weight_decay=args.weight_decay))
    tx = optax.chain(*tx_parts)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, vq_state=vq_state)
    return model, state


def make_train_step(model):
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            variables = {"params": params, "vq_state": state.vq_state}
            out, updates = model.apply(variables, batch, training=True, mutable=["vq_state"])
            metrics = {
                "total": out["total_loss"],
                "encoder": out["encoder_loss"],
                "recon": out["recon_loss"],
                "vq": out["vq_loss_state"],
                "layer_w0": out["layer_weights"][0],
                "layer_w1": out["layer_weights"][1] if out["layer_weights"].shape[0] > 1 else out["layer_weights"][0],
            }
            return out["total_loss"], (metrics, updates["vq_state"])

        (loss, (metrics, vq_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads).replace(vq_state=vq_state)
        return state, metrics

    return train_step


def make_eval_step(model):
    @jax.jit
    def eval_step(params, vq_state, batch):
        out = model.apply({"params": params, "vq_state": vq_state}, batch, training=False)
        return out["recon_loss"], out["indices"]

    return eval_step


def _flatten_indices(indices: np.ndarray, codebook_size: int) -> np.ndarray:
    flat = np.zeros((indices.shape[0],), dtype=np.int64)
    for i in range(indices.shape[1]):
        flat = flat * codebook_size + indices[:, i].astype(np.int64)
    return flat


def evaluate(model, params, vq_state, actions, batch_size):
    eval_step = make_eval_step(model)
    total_loss = 0.0
    total_n = 0
    all_indices = []
    rng = np.random.RandomState(0)
    for batch in iterate_minibatches(actions, batch_size, shuffle=False, drop_last=False, rng=rng):
        recon_loss, indices = eval_step(params, vq_state, jnp.asarray(batch))
        total_loss += float(recon_loss) * batch.shape[0]
        total_n += batch.shape[0]
        all_indices.append(np.asarray(indices))
    return total_loss / max(total_n, 1), np.concatenate(all_indices, axis=0)


def code_usage(indices, codebook_size, num_layers):
    total_codes = codebook_size ** num_layers
    combos = _flatten_indices(indices, codebook_size)
    combo_counts = np.bincount(combos, minlength=total_codes)
    probs = combo_counts.astype(np.float64) / max(combo_counts.sum(), 1)
    active_probs = probs[probs > 0]
    perplexity = float(np.exp(-(active_probs * np.log(active_probs)).sum())) if active_probs.size else 0.0
    per_layer = []
    for layer in range(num_layers):
        counts = np.bincount(indices[:, layer], minlength=codebook_size)
        per_layer.append(counts.astype(int).tolist())
    return {
        "num_combinations": int(total_codes),
        "num_used_combinations": int((combo_counts > 0).sum()),
        "combo_counts": combo_counts.astype(int).tolist(),
        "perplexity": perplexity,
        "per_layer_counts": per_layer,
    }


def save_checkpoint(state, step, output_dir, args, train_actions, val_metrics, history):
    ckpt_dir = os.path.join(output_dir, "jax_checkpoints")
    payload = {
        "params": state.params,
        "vq_state": state.vq_state,
        "opt_state": state.opt_state,
        "step": int(step),
        "backend": "jax_vqvae",
        "model_type": "hand_vqvae",
        "model_args": build_model_args(args),
        "train_args": vars(args),
        "action_min": train_actions.min(axis=0).astype(np.float32),
        "action_max": train_actions.max(axis=0).astype(np.float32),
        "action_mean": train_actions.mean(axis=0).astype(np.float32),
        "action_std": train_actions.std(axis=0).clip(min=1e-6).astype(np.float32),
        "final_metrics": val_metrics,
        "history": history,
    }
    return save_jax_checkpoint(ckpt_dir, payload, step)


def save_training_curves(history, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping curves")
        return

    if not history["steps"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["steps"], history["train_total"], label="train total")
    axes[0].set_title("Total")
    axes[1].plot(history["steps"], history["train_recon"], label="train recon")
    if history["eval_steps"]:
        axes[1].plot(history["eval_steps"], history["val_recon"], "o-", label="val recon")
    axes[1].set_title("Recon MSE")
    axes[1].legend()
    axes[2].plot(history["steps"], history["train_encoder"], label="weighted L1")
    axes[2].plot(history["steps"], history["train_vq"], label="vq")
    axes[2].set_title("DQ-RISE Losses")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_curves_jax.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {path}")


def save_summary(output_dir, args, history, final_metrics, usage, train_actions):
    summary = {
        "backend": "jax_vqvae",
        "model_type": "hand_vqvae",
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "output_dir": args.output_dir,
        "model_args": build_model_args(args),
        "train_args": vars(args),
        "action_min": train_actions.min(axis=0).astype(float).tolist(),
        "action_max": train_actions.max(axis=0).astype(float).tolist(),
        "final_metrics": final_metrics,
        "code_usage": usage,
        "history": history,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    train_actions = load_hand_action_array(args.train_dir)
    test_actions = load_hand_action_array(args.test_dir)

    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)
    model, state = create_train_state(args, lr_schedule)
    print(f"Model parameters: {count_params(state.params):,}")
    print("DQ-RISE strict VQ-VAE: EMA ResidualVQ, weighted L1 x3 + VQ x5")
    print(
        f"Codebook: {args.num_vq_layers} layers x {args.codebook_size} entries = "
        f"{args.codebook_size ** args.num_vq_layers} combinations"
    )

    train_step = make_train_step(model)
    history = {
        "steps": [],
        "train_total": [],
        "train_encoder": [],
        "train_recon": [],
        "train_vq": [],
        "train_layer_w0": [],
        "train_layer_w1": [],
        "train_lr": [],
        "eval_steps": [],
        "val_recon": [],
    }
    train_iter = iterate_minibatches(train_actions, args.batch_size, shuffle=True, drop_last=True, rng=rng)
    t0 = time.time()

    for step in range(args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iterate_minibatches(train_actions, args.batch_size, shuffle=True, drop_last=True, rng=rng)
            batch = next(train_iter)

        state, metrics = train_step(state, jnp.asarray(batch))
        history["steps"].append(int(step))
        history["train_total"].append(float(metrics["total"]))
        history["train_encoder"].append(float(metrics["encoder"]))
        history["train_recon"].append(float(metrics["recon"]))
        history["train_vq"].append(float(metrics["vq"]))
        history["train_layer_w0"].append(float(metrics["layer_w0"]))
        history["train_layer_w1"].append(float(metrics["layer_w1"]))
        history["train_lr"].append(float(lr_schedule[step]))

        if step % args.print_freq == 0:
            print(
                f"[Step {step:>5d}] total={float(metrics['total']):.6f} "
                f"encoder={float(metrics['encoder']):.6f} "
                f"recon={float(metrics['recon']):.6f} "
                f"vq={float(metrics['vq']):.6f} "
                f"w=({float(metrics['layer_w0']):.3f},{float(metrics['layer_w1']):.3f}) "
                f"lr={lr_schedule[step]:.2e} ({time.time() - t0:.0f}s)"
            )

        if step > 0 and step % args.eval_freq == 0:
            val_recon, val_indices = evaluate(model, state.params, state.vq_state, test_actions, args.batch_size)
            history["eval_steps"].append(int(step))
            history["val_recon"].append(float(val_recon))
            usage = code_usage(val_indices, args.codebook_size, args.num_vq_layers)
            print(
                f"           val_recon={val_recon:.6f} "
                f"used={usage['num_used_combinations']}/{usage['num_combinations']} "
                f"ppl={usage['perplexity']:.2f}"
            )

        if step > 0 and step % args.save_freq == 0:
            val_recon, val_indices = evaluate(model, state.params, state.vq_state, test_actions, args.batch_size)
            usage = code_usage(val_indices, args.codebook_size, args.num_vq_layers)
            metrics_payload = {"val_recon_mse": float(val_recon), "code_usage": usage}
            saved = save_checkpoint(state, step, args.output_dir, args, train_actions, metrics_payload, history)
            print(f"           checkpoint saved -> {saved}")

    val_recon, val_indices = evaluate(model, state.params, state.vq_state, test_actions, args.batch_size)
    usage = code_usage(val_indices, args.codebook_size, args.num_vq_layers)
    final_metrics = {"val_recon_mse": float(val_recon)}
    final_ckpt = save_checkpoint(state, args.total_steps, args.output_dir, args, train_actions, final_metrics, history)
    save_training_curves(history, args.output_dir)
    save_summary(args.output_dir, args, history, final_metrics, usage, train_actions)

    print(
        f"\nFinal val_recon={val_recon:.6f} "
        f"used={usage['num_used_combinations']}/{usage['num_combinations']} "
        f"ppl={usage['perplexity']:.2f}"
    )
    print(f"Done! JAX checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()

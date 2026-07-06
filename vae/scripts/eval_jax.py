"""
Evaluate Hand Action VAE checkpoints saved by the JAX training pipeline.

Usage:
    conda run -n serl python vae/scripts/eval_jax.py \
        --ckpt_dir outputs/hand_vae_jax/jax_checkpoints \
        --test_dir data/example_task/demos/success/test \
        --free_run --num_samples 20 --save_plot
"""

import argparse
import glob
import importlib.util
import os

import jax
import numpy as np

_vae_root = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(_vae_root, "..")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_vae_root, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HandActionVAE = _load("model/hand_vae_jax.py", "hand_vae_jax").HandActionVAE
load_hand_actions = _load("model/data_io.py", "data_io").load_hand_actions
restore_jax_checkpoint = _load("model/checkpoint_compat.py", "checkpoint_compat").restore_jax_checkpoint

JOINT_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
DEFAULT_INIT = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def load_trajectory(path):
    return load_hand_actions(path).numpy()


def make_predict_fn(model, deterministic):
    if deterministic:
        @jax.jit
        def predict_fn(params, window, rng):
            del rng
            return model.apply({"params": params}, window, deterministic=True, method=HandActionVAE.predict)
    else:
        @jax.jit
        def predict_fn(params, window, rng):
            return model.apply(
                {"params": params},
                window,
                deterministic=False,
                method=HandActionVAE.predict,
                rngs={"sample": rng},
            )

    return predict_fn


def rollout_teacher_forcing(model, params, gt_actions, window_size):
    predict_fn = make_predict_fn(model, deterministic=True)
    pred = gt_actions.copy()
    rng = jax.random.PRNGKey(0)
    for step in range(window_size, gt_actions.shape[0]):
        window = gt_actions[step - window_size : step][None, ...]
        pred[step] = np.asarray(predict_fn(params, window, rng))[0]
    return pred


def rollout_autoregressive(model, params, seed, total_steps, window_size, deterministic, rng_seed):
    predict_fn = make_predict_fn(model, deterministic=deterministic)
    if seed.shape[0] < window_size:
        pad = np.repeat(seed[0:1], window_size - seed.shape[0], axis=0)
        buffer = np.concatenate([pad, seed], axis=0)
    else:
        buffer = seed.copy()

    all_actions = [row.copy() for row in buffer]
    rng = jax.random.PRNGKey(rng_seed)
    for _ in range(total_steps - len(all_actions)):
        rng, sample_key = jax.random.split(rng)
        window = np.stack(all_actions[-window_size:])[None, ...]
        pred = np.asarray(predict_fn(params, window, sample_key))[0]
        all_actions.append(pred)

    return np.stack(all_actions[:total_steps], axis=0)


def eval_gt_comparison(model, params, traj_path, window_size, deterministic, num_samples, verbose):
    gt = load_trajectory(traj_path)
    traj_id = os.path.basename(traj_path).split("trajectory_")[1].split("_")[0]
    pred_tf = rollout_teacher_forcing(model, params, gt, window_size)
    mse_tf = ((pred_tf - gt) ** 2).mean(axis=1)

    ar_runs = []
    for sample_idx in range(num_samples):
        ar_runs.append(
            rollout_autoregressive(
                model,
                params,
                gt[:window_size],
                gt.shape[0],
                window_size,
                deterministic,
                rng_seed=sample_idx,
            )
        )
    ar_runs = np.stack(ar_runs, axis=0)
    ar_mean = ar_runs.mean(axis=0)
    ar_std = ar_runs.std(axis=0)
    mse_ar_mean = ((ar_mean - gt) ** 2).mean(axis=1)
    copy_mse = ((gt[window_size:] - gt[window_size - 1 : -1]) ** 2).mean().item()

    if verbose:
        print(f"\nTrajectory {traj_id}: {gt.shape[0]} steps (seed={window_size}, predict={gt.shape[0] - window_size})")
        print(f"  Teacher Forcing MSE:         {mse_tf[window_size:].mean():.6f}")
        print(f"  Autoregressive MSE (mean):   {mse_ar_mean[window_size:].mean():.6f}  ({num_samples} samples)")
        print(f"  Copy Baseline MSE:           {copy_mse:.6f}")
        if num_samples > 1:
            print(f"  AR prediction std (mean):    {ar_std[window_size:].mean():.6f}")

    return {
        "traj_id": traj_id,
        "gt": gt,
        "pred_tf": pred_tf,
        "mse_tf": mse_tf,
        "ar_runs": ar_runs,
        "ar_mean": ar_mean,
        "ar_std": ar_std,
        "mse_ar_mean": mse_ar_mean,
        "copy_mse": copy_mse,
        "num_samples": num_samples,
    }


def eval_free_run(model, params, seed, max_steps, window_size, deterministic, num_samples, label="free"):
    runs = []
    for sample_idx in range(num_samples):
        runs.append(
            rollout_autoregressive(
                model,
                params,
                seed,
                max_steps,
                window_size,
                deterministic,
                rng_seed=sample_idx,
            )
        )
    runs = np.stack(runs, axis=0)
    print(f"\nFree-run: {max_steps} steps, {num_samples} samples")
    print(f"  Seed: {seed[0].round(3)}")
    print(f"  Final state (mean): {runs[:, -1, :].mean(axis=0).round(3)}")
    print(f"  Final state (std):  {runs[:, -1, :].std(axis=0).round(3)}")
    return {
        "label": label,
        "seed": seed,
        "runs": runs,
        "num_samples": num_samples,
        "max_steps": max_steps,
    }


def plot_gt_comparison(result, output_dir, window_size):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"Trajectory {result['traj_id']} comparison ({result['num_samples']} AR samples)",
        fontsize=14,
    )

    gt = result["gt"]
    pred_tf = result["pred_tf"]
    ar_mean = result["ar_mean"]
    ar_std = result["ar_std"]
    steps = np.arange(gt.shape[0])

    for idx, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        ax.plot(steps, gt[:, idx], color="black", linewidth=2, label="GT")
        ax.plot(steps, pred_tf[:, idx], color="green", linewidth=1.2, label="Teacher forcing")
        ax.plot(steps, ar_mean[:, idx], color="royalblue", linewidth=1.5, label="AR mean")
        if result["num_samples"] > 1:
            ax.fill_between(
                steps,
                ar_mean[:, idx] - ar_std[:, idx],
                ar_mean[:, idx] + ar_std[:, idx],
                color="royalblue",
                alpha=0.2,
            )
        ax.axvline(x=window_size - 0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    path = os.path.join(output_dir, f"traj_{result['traj_id']}_comparison_jax.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_free_run(result, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    runs = result["runs"]
    num_samples, max_steps, _ = runs.shape
    seed_len = result["seed"].shape[0]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"Free-run: {max_steps} steps, {num_samples} stochastic samples  (seed={seed_len} frames)",
        fontsize=14,
    )

    mean = runs.mean(axis=0)
    std = runs.std(axis=0)

    for idx, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        for sample_idx in range(min(num_samples, 30)):
            ax.plot(np.arange(max_steps), runs[sample_idx, :, idx], color="steelblue", alpha=0.2, linewidth=0.8)
        ax.plot(np.arange(max_steps), mean[:, idx], "b-", label=f"Mean (n={num_samples})", linewidth=2)
        if num_samples > 1:
            ax.fill_between(
                np.arange(max_steps),
                mean[:, idx] - std[:, idx],
                mean[:, idx] + std[:, idx],
                color="steelblue",
                alpha=0.2,
            )
        ax.axvline(x=seed_len - 0.5, color="gray", linestyle=":", alpha=0.4, label="Seed boundary")
        ax.set_ylabel(name)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    path = os.path.join(output_dir, f"free_run_{result['label']}_jax.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAX Hand Action VAE")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(_proj_root, "outputs/hand_vae_jax/jax_checkpoints"))
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join(_proj_root, "data/example_task/demos/success/test"),
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(_proj_root, "visualizations/vae_eval_jax"))
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--free_run", action="store_true")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--traj_id", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--init_state", type=float, nargs=6, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    payload = restore_jax_checkpoint(args.ckpt_dir, step=args.step)
    model_args = payload["model_args"]
    params = payload["params"]
    model = HandActionVAE(**model_args)

    if model_args["window_size"] != args.window_size:
        print(f"  Note: overriding --window_size {args.window_size} -> {model_args['window_size']} (from ckpt)")
        args.window_size = model_args["window_size"]

    print(f"Model: {args.ckpt_dir}")
    print(
        f"  Arch: action_dim={model_args['action_dim']}  "
        f"window_size={model_args['window_size']}  "
        f"hidden_dim={model_args['hidden_dim']}  "
        f"latent_dim={model_args['latent_dim']}  "
        f"encoder={model_args['encoder_type']}  "
        f"depth={model_args['num_hidden_layers']}  "
        f"aux_head={'yes' if model_args.get('recon_aux_weight', 0) > 0 else 'no'}"
    )
    print(f"Sampling: {'deterministic (mu)' if args.deterministic else f'stochastic ({args.num_samples} samples)'}")

    if args.free_run:
        if args.init_state:
            seed = np.asarray(args.init_state, dtype=np.float32)[None, :]
            label = "custom"
        elif args.traj_id:
            all_files = sorted(glob.glob(os.path.join(args.test_dir, "trajectory_*_demo_expert.pt")))
            matches = [path for path in all_files if f"trajectory_{args.traj_id[0]}_" in path]
            if matches:
                seed = load_trajectory(matches[0])[: args.window_size]
                label = f"seed_traj{args.traj_id[0]}"
            else:
                seed = DEFAULT_INIT[None, :]
                label = "default"
        else:
            seed = DEFAULT_INIT[None, :]
            label = "default"

        result = eval_free_run(
            model,
            params,
            seed,
            args.max_steps,
            args.window_size,
            args.deterministic,
            args.num_samples,
            label=label,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"free_run_{label}_jax.npz")
        np.savez(save_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        print(f"  Trajectory data saved: {save_path}")
        if args.save_plot:
            plot_free_run(result, args.output_dir)
        return

    all_files = sorted(glob.glob(os.path.join(args.test_dir, "trajectory_*_demo_expert.pt")))
    if args.traj_id:
        files = []
        for traj_id in args.traj_id:
            files.extend([path for path in all_files if f"trajectory_{traj_id}_" in path])
    elif args.all:
        files = all_files
    else:
        files = all_files[:3]

    if not files:
        print("No trajectory files found!")
        return

    results = []
    for path in files:
        result = eval_gt_comparison(
            model,
            params,
            path,
            args.window_size,
            args.deterministic,
            args.num_samples,
            verbose=True,
        )
        results.append(result)
        if args.save_plot:
            plot_gt_comparison(result, args.output_dir, args.window_size)

    print(f"\n{'=' * 60}")
    print(f"Summary ({len(results)} trajectories, {args.num_samples} samples each)")
    print(f"{'=' * 60}")
    for result in results:
        tf_mse = result["mse_tf"][args.window_size :].mean()
        ar_mse = result["mse_ar_mean"][args.window_size :].mean()
        print(
            f"  traj {result['traj_id']:>4}: TF={tf_mse:.6f}  AR={ar_mse:.6f}  "
            f"copy={result['copy_mse']:.6f}  AR_std={result['ar_std'][args.window_size :].mean():.4f}"
        )


if __name__ == "__main__":
    main()

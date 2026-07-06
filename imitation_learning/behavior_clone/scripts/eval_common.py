"""Shared trajectory loading and plotting helpers for JAX behavior_clone eval."""

import os

import numpy as np
import torch


ARM_NAMES = [f"arm_{i}" for i in range(6)]
HAND_NAMES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ARM_NAMES + HAND_NAMES


def load_trajectory(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    actions = data["actions"][:, 0, :].float()
    main = data["curr_obs"]["main_images"][:, 0].permute(0, 3, 1, 2).contiguous()
    extra = data["curr_obs"]["extra_view_images"][:, 0, 0].permute(0, 3, 1, 2).contiguous()
    traj_id = os.path.basename(path).split("trajectory_")[1].split("_")[0]
    return {
        "traj_id": traj_id,
        "actions": actions,
        "imgs_main": main,
        "imgs_extra": extra,
        "T": int(actions.shape[0]),
    }


def detect_grasp_onset(actions, threshold=0.02, lookahead=3, min_count=2):
    if actions.shape[0] <= 1:
        return None
    delta = actions[1:, 6:] - actions[:-1, 6:]
    norm = np.linalg.norm(delta, axis=1)
    delta_norm = np.concatenate([norm, np.zeros((1,))])
    for t in range(actions.shape[0] - 1):
        window = delta_norm[t:min(actions.shape[0] - 1, t + lookahead)]
        if int((window > threshold).sum()) >= min_count:
            return int(t)
    return None


def plot_trajectory_actions(
    traj_id,
    T,
    gt_target,
    ar_runs,
    no_corr,
    num_samples,
    output_dir,
    onset_step=None,
    xlim_max=None,
    xtick_step=5,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_mean = ar_runs.mean(axis=0)
    ar_std = ar_runs.std(axis=0)
    steps = np.arange(T)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)
    fig.suptitle(f"Trajectory {traj_id}:  AR rollout  (T={T}, n={num_samples})", fontsize=14)

    for i, name in enumerate(JOINT_NAMES):
        ax = axes.flat[i]
        for s in range(num_samples):
            ax.plot(steps, ar_runs[s, :, i], color="steelblue", alpha=0.25, linewidth=0.8)
        ax.plot(steps, ar_mean[:, i], color="royalblue", linewidth=2.0, label=f"AR mean (n={num_samples})")
        ax.fill_between(
            steps,
            ar_mean[:, i] - ar_std[:, i],
            ar_mean[:, i] + ar_std[:, i],
            color="steelblue",
            alpha=0.15,
        )
        ax.plot(steps, gt_target[:, i], "k-", linewidth=2.0, label="GT next-pose", zorder=10)
        ax.plot(
            steps,
            no_corr[:, i],
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="No correction (VAE prior)",
        )
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)
            ax.set_xticks(np.arange(0, xlim_max + 1, xtick_step))
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Decision step t", fontsize=10)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_actions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_mse(traj_id, T, gt_target, ar_runs, no_corr, output_dir, onset_step=None, xlim_max=None, xtick_step=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_arm_mse = ((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean(axis=2).mean(axis=0)
    ar_hand_mse = ((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean(axis=2).mean(axis=0)
    nc_arm_mse = ((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean(axis=1)
    nc_hand_mse = ((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean(axis=1)
    copy_arm = np.concatenate([((gt_target[:-1, :6] - gt_target[1:, :6]) ** 2).mean(axis=1), np.zeros(1)])
    copy_hand = np.concatenate([((gt_target[:-1, 6:] - gt_target[1:, 6:]) ** 2).mean(axis=1), np.zeros(1)])
    steps = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Trajectory {traj_id}: Per-step MSE", fontsize=13)

    for ax, side, ar_mse, nc_mse, cp_mse in [
        (axes[0], "Arm", ar_arm_mse, nc_arm_mse, copy_arm),
        (axes[1], "Hand", ar_hand_mse, nc_hand_mse, copy_hand),
    ]:
        ax.plot(steps, ar_mse, color="royalblue", linewidth=1.5, label=f"AR (mean={ar_mse.mean():.5f})")
        ax.plot(steps, nc_mse, color="gray", linestyle="--", linewidth=1.2, label=f"No-corr (mean={nc_mse.mean():.5f})")
        ax.plot(steps, cp_mse, "k:", linewidth=1.0, alpha=0.5, label=f"Copy baseline (mean={cp_mse.mean():.5f})")
        if onset_step is not None:
            ax.axvline(onset_step, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Decision step t")
        ax.set_ylabel("MSE")
        ax.set_title(f"{side} MSE")
        ax.grid(True, alpha=0.3)
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)
            ax.set_xticks(np.arange(0, xlim_max + 1, xtick_step))
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    path = os.path.join(output_dir, f"traj_{traj_id}_ar_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_results, output_dir, num_samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"BC AR Evaluation Summary  ({len(all_results)} trajectories, n={num_samples})", fontsize=13)
    for ax, side in [(axes[0], "arm"), (axes[1], "hand")]:
        ar_v = np.mean([r[f"ar_{side}_mse"] for r in all_results])
        nc_v = np.mean([r[f"nc_{side}_mse"] for r in all_results])
        cp_v = np.mean([r[f"copy_{side}_mse"] for r in all_results])
        labels = ["AR", "No Correction", "Copy Baseline"]
        values = [ar_v, nc_v, cp_v]
        colors = ["royalblue", "gray", "black"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean MSE")
        ax.set_title(f"{side.title()} MSE")
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.0)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "summary_ar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot: {path}")


def plot_per_trajectory_bar(all_results, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_results = sorted(all_results, key=lambda r: r["ar_hand_mse"])
    traj_ids = [r["traj_id"] for r in sorted_results]
    ar_hand = [r["ar_hand_mse"] for r in sorted_results]
    nc_hand = [r["nc_hand_mse"] for r in sorted_results]

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(traj_ids) * 0.5), 5))
    x = np.arange(len(traj_ids))
    width = 0.35
    ax.bar(x - width / 2, ar_hand, width, color="royalblue", alpha=0.8, label="AR hand")
    ax.bar(x + width / 2, nc_hand, width, color="gray", alpha=0.6, label="No-corr hand")
    ax.set_xticks(x)
    ax.set_xticklabels(traj_ids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Hand MSE")
    ax.set_title("Per-trajectory AR hand MSE (sorted)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, "per_trajectory_hand_mse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-trajectory bar: {path}")


def plot_latent_diagnostics(all_results, output_dir, num_samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Latent Diagnostics  ({len(all_results)} trajectories, n={num_samples})", fontsize=13)

    all_dz_norm, all_mu_norm, all_dz_d0, all_dz_d1 = [], [], [], []
    for result in all_results:
        dz, mu = result["delta_z"], result["mu_prior"]
        all_dz_norm.extend(np.linalg.norm(dz, axis=-1).flatten().tolist())
        all_mu_norm.extend(np.linalg.norm(mu, axis=-1).flatten().tolist())
        all_dz_d0.extend(dz[:, :, 0].flatten().tolist())
        if dz.shape[-1] > 1:
            all_dz_d1.extend(dz[:, :, 1].flatten().tolist())

    ax = axes[0, 0]
    ax.hist(all_dz_norm, bins=80, color="steelblue", alpha=0.7, density=True)
    ax.axvline(np.mean(all_dz_norm), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(all_dz_norm):.4f}")
    ax.set_xlabel("|delta_z|")
    ax.set_title("delta_z magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(all_mu_norm, bins=80, color="gray", alpha=0.7, density=True)
    ax.axvline(np.mean(all_mu_norm), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(all_mu_norm):.4f}")
    ax.set_xlabel("|mu_prior|")
    ax.set_title("mu_prior magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if all_dz_d1:
        ax.scatter(all_dz_d0[:5000], all_dz_d1[:5000], s=2, alpha=0.3, color="steelblue")
        ax.set_xlabel("delta_z[0]")
        ax.set_ylabel("delta_z[1]")
        ax.set_title("delta_z scatter (first 5k)")
    else:
        ax.hist(all_dz_d0, bins=80, color="steelblue", alpha=0.7)
        ax.set_xlabel("delta_z[0]")
        ax.set_title("delta_z[0] distribution")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    max_T = max(r["T"] for r in all_results)
    dz_by_t = [[] for _ in range(max_T)]
    for result in all_results:
        norms = np.linalg.norm(result["delta_z"], axis=-1)
        for t in range(result["T"]):
            dz_by_t[t].extend(norms[:, t].tolist())
    mean_t = [np.mean(v) if v else 0 for v in dz_by_t]
    std_t = [np.std(v) if v else 0 for v in dz_by_t]
    ax.plot(mean_t, color="royalblue", linewidth=1.5, label="mean |delta_z|")
    ax.fill_between(
        range(len(mean_t)),
        np.array(mean_t) - np.array(std_t),
        np.array(mean_t) + np.array(std_t),
        color="steelblue",
        alpha=0.2,
    )
    ax.set_xlabel("Decision step t")
    ax.set_ylabel("|delta_z|")
    ax.set_title("delta_z magnitude over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "latent_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latent diagnostics: {path}")

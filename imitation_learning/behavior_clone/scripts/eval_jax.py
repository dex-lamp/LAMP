"""Direct JAX evaluation for the integrated single-step behavior_clone policy."""

import argparse
import glob
import importlib.util
import json
import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dataset_mod = _load(os.path.join(_BC_ROOT, "model/bc_dataset.py"), "bc_dataset_behavior_clone_eval_jax")
_policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy_jax.py"), "bc_policy_behavior_clone_eval_jax")
_compat_mod = _load(os.path.join(_BC_ROOT, "model/checkpoint_compat.py"), "bc_ckpt_compat_behavior_clone_eval_jax")
_eval_common_mod = _load(os.path.join(_BC_ROOT, "scripts/eval_common.py"), "bc_eval_common_behavior_clone")

BCDataset = _dataset_mod.BCDataset
BCPolicy = _policy_mod.BCPolicy
resolve_jax_checkpoint_dir = _compat_mod.resolve_jax_checkpoint_dir
restore_jax_checkpoint = _compat_mod.restore_jax_checkpoint
load_trajectory = _eval_common_mod.load_trajectory
detect_grasp_onset = _eval_common_mod.detect_grasp_onset
plot_trajectory_actions = _eval_common_mod.plot_trajectory_actions
plot_trajectory_mse = _eval_common_mod.plot_trajectory_mse
plot_summary = _eval_common_mod.plot_summary
plot_per_trajectory_bar = _eval_common_mod.plot_per_trajectory_bar
plot_latent_diagnostics = _eval_common_mod.plot_latent_diagnostics


def get_args():
    p = argparse.ArgumentParser(description="Evaluate integrated JAX behavior_clone policy")
    p.add_argument("--ckpt_dir", type=str, required=True, help="JAX checkpoint root or jax_checkpoints dir")
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_trajs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable_image_random_crop",
        action="store_true",
        help="Enable paired edge-padded random crop during eval. Disabled by default.",
    )
    p.add_argument(
        "--random_crop_padding",
        type=int,
        default=None,
        help=(
            "Image random crop padding used only with --enable_image_random_crop. "
            "Defaults to checkpoint train_args.random_crop_padding, then 4."
        ),
    )
    p.add_argument(
        "--hand_history_std_floor",
        type=float,
        default=None,
        help=(
            "Override checkpoint hand-history std floor for direct normalized "
            "hand-history heads. Defaults to checkpoint metadata, then 1e-6."
        ),
    )
    p.add_argument("--debug_dir", type=str, default=None, help="Optional directory for per-trajectory latent/action debug npz files.")
    return p.parse_args()


def torch_batch_to_numpy(batch):
    return {k: np.asarray(v.numpy(), dtype=np.float32) for k, v in batch.items()}


def resolve_random_crop_padding(args, payload) -> int:
    if not args.enable_image_random_crop:
        return 0
    if args.random_crop_padding is not None:
        padding = int(args.random_crop_padding)
    else:
        padding = int(payload.get("train_args", {}).get("random_crop_padding", 4))
    if padding < 0:
        raise ValueError(f"--random_crop_padding must be >= 0, got {padding}")
    return padding


def resolve_hand_history_std_floor(args, payload) -> float:
    if args.hand_history_std_floor is not None:
        floor = float(args.hand_history_std_floor)
    else:
        model_args = payload.get("model_args", {})
        train_args = payload.get("train_args", {})
        floor = float(model_args.get("hand_history_std_floor", train_args.get("hand_history_std_floor", 1e-6)))
    if floor < 0:
        raise ValueError(f"--hand_history_std_floor must be >= 0, got {floor}")
    return floor


def _edge_random_crop_nchw_np(img: np.ndarray, crop_y: int, crop_x: int, padding: int) -> np.ndarray:
    if padding <= 0:
        return img
    _, height, width = img.shape
    padded = np.pad(
        img,
        ((0, 0), (padding, padding), (padding, padding)),
        mode="edge",
    )
    return np.ascontiguousarray(padded[:, crop_y : crop_y + height, crop_x : crop_x + width])


def paired_edge_random_crop_nchw_np(
    img_main: np.ndarray,
    img_extra: np.ndarray,
    crop_key,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paired edge-padded random crop to NCHW images when enabled."""
    if padding <= 0:
        return img_main, img_extra
    if img_main.shape[0] != img_extra.shape[0]:
        raise ValueError(f"Image batch sizes differ: {img_main.shape} vs {img_extra.shape}")
    crop_from = np.asarray(
        jax.random.randint(crop_key, (img_main.shape[0], 2), 0, 2 * padding + 1)
    )
    main_out = []
    extra_out = []
    for idx, (crop_y, crop_x) in enumerate(crop_from):
        crop_y = int(crop_y)
        crop_x = int(crop_x)
        main_out.append(_edge_random_crop_nchw_np(img_main[idx], crop_y, crop_x, padding))
        extra_out.append(_edge_random_crop_nchw_np(img_extra[idx], crop_y, crop_x, padding))
    return np.stack(main_out, axis=0), np.stack(extra_out, axis=0)


def build_past_window_np(hand_seq: np.ndarray, t: int, window_size: int) -> np.ndarray:
    start = t - window_size + 1
    if start < 0:
        pad_len = -start
        pad = np.repeat(hand_seq[0:1], pad_len, axis=0)
        return np.concatenate([pad, hand_seq[0 : t + 1]], axis=0)
    return hand_seq[start : t + 1]


def normalize_hand_window_np(
    hand_window: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    hand_history_std_floor: float,
) -> np.ndarray:
    hand_mean = action_mean[6:12]
    hand_std = np.maximum(action_std[6:12], max(float(hand_history_std_floor), 1e-6))
    return ((hand_window - hand_mean) / hand_std).astype(np.float32)


def policy_hand_window_from_batch(batch_np, hand_prior_source: str):
    if hand_prior_source in {"mlp_direct", "decoder_only", "vq_codebook"}:
        return batch_np["past_hand_state_win"]
    if hand_prior_source == "pca_raw":
        return batch_np["past_hand_win_raw"]
    return batch_np["past_hand_win"]


def policy_hand_window_from_raw(
    hand_prior_source: str,
    raw_window: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    hand_history_std_floor: float,
) -> np.ndarray:
    if hand_prior_source in {"mlp_direct", "decoder_only", "vq_codebook"}:
        return normalize_hand_window_np(raw_window, action_mean, action_std, hand_history_std_floor)
    return raw_window.astype(np.float32)


def load_policy_and_metadata(ckpt_root: str):
    ckpt_dir = resolve_jax_checkpoint_dir(ckpt_root)
    payload = restore_jax_checkpoint(ckpt_dir)
    model_args = payload["model_args"]
    hand_codebook = None
    if model_args.get("hand_prior_source") == "vq_codebook":
        if "hand_codebook" in payload:
            hand_codebook = np.asarray(payload["hand_codebook"], dtype=np.float32)
        else:
            hand_codebook = np.load(model_args["hand_codebook_path"]).astype(np.float32)
    hand_pca_mean = None
    hand_pca_components = None
    if model_args.get("hand_prior_source") == "pca_raw":
        if "hand_pca_mean" not in payload or "hand_pca_components" not in payload:
            raise ValueError(
                "pca_raw checkpoint must embed payload['hand_pca_mean'] and "
                "payload['hand_pca_components']."
            )
        hand_pca_mean = np.asarray(payload["hand_pca_mean"], dtype=np.float32)
        hand_pca_components = np.asarray(payload["hand_pca_components"], dtype=np.float32)
    model = BCPolicy(
        vae_model_args=model_args.get("vae"),
        hand_prior_source=model_args.get("hand_prior_source", "vae"),
        freeze_vae=model_args.get("freeze_vae", True),
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
        hand_pca_dim=int(model_args.get("hand_pca_dim", 2)),
        window_size=int(model_args.get("window_size", 8)),
        backbone_type=model_args.get("backbone_type", "hf_resnet18"),
        backbone_config=model_args.get("backbone_config"),
        state_encoder_type=model_args["state_encoder"],
        arm_state_dim=model_args.get("arm_state_dim", 6),
        hand_state_dim=model_args.get("hand_state_dim", 6),
        dropout=model_args["dropout"],
        hil_serl_root=model_args.get("hil_serl_root", getattr(_policy_mod, "DEFAULT_HIL_SERL_ROOT", "")),
        hil_serl_pooling_method=model_args.get("hil_serl_pooling_method", "spatial_learned_embeddings"),
        hil_serl_num_spatial_blocks=model_args.get("hil_serl_num_spatial_blocks", 8),
        hil_serl_bottleneck_dim=model_args.get("hil_serl_bottleneck_dim", 256),
    )
    return model, payload["params"], payload["batch_stats"], payload


def _make_variables(params, batch_stats):
    variables = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats
    return variables


def make_eval_step(model):
    @partial(jax.jit, static_argnames=("zero_delta",))
    def eval_step(params, batch_stats, img_main, img_extra, state_in, past_hand_win, zero_delta=False):
        return model.apply(
            _make_variables(params, batch_stats),
            img_main,
            img_extra,
            state_in,
            past_hand_win,
            zero_delta=zero_delta,
            deterministic=True,
            train_backbone=False,
        )

    return eval_step


def make_dropout_step(model):
    @jax.jit
    def dropout_step(params, batch_stats, img_main, img_extra, state_in, past_hand_win, dropout_key):
        return model.apply(
            _make_variables(params, batch_stats),
            img_main,
            img_extra,
            state_in,
            past_hand_win,
            deterministic=False,
            train_backbone=False,
            rngs={"dropout": dropout_key},
        )

    return dropout_step


def evaluate_validation_mse(
    model,
    params,
    batch_stats,
    test_dir,
    batch_size,
    action_mean,
    action_std,
    window_size,
    hand_prior_source,
    hand_codebook=None,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 0,
    hand_history_std_floor: float = 1e-6,
):
    action_mean_t = np.asarray(action_mean, dtype=np.float32)
    action_std_t = np.asarray(action_std, dtype=np.float32)
    test_ds = BCDataset(
        test_dir,
        action_mean=torch.tensor(action_mean_t),
        action_std=torch.tensor(action_std_t),
        window_size=window_size,
        noise_std_hand=0.0,
        noise_std_arm=0.0,
        hand_codebook=hand_codebook,
        enable_image_random_crop=enable_image_random_crop,
        random_crop_padding=random_crop_padding,
        hand_history_std_floor=hand_history_std_floor,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    eval_step = make_eval_step(model)
    sums = {"arm": 0.0, "hand_full": 0.0, "hand_no_corr": 0.0, "hand_index": 0.0, "hand_index_acc": 0.0, "total": 0.0, "n": 0}

    for batch in test_loader:
        batch_np = torch_batch_to_numpy(batch)
        out = eval_step(
            params,
            batch_stats,
            jnp.asarray(batch_np["img_main"]),
            jnp.asarray(batch_np["img_extra"]),
            jnp.asarray(batch_np["state"]),
            jnp.asarray(policy_hand_window_from_batch(batch_np, hand_prior_source)),
            zero_delta=False,
        )
        out_zero = eval_step(
            params,
            batch_stats,
            jnp.asarray(batch_np["img_main"]),
            jnp.asarray(batch_np["img_extra"]),
            jnp.asarray(batch_np["state"]),
            jnp.asarray(policy_hand_window_from_batch(batch_np, hand_prior_source)),
            zero_delta=True,
        )
        gt = batch_np["gt_action"]
        arm_loss = float(np.mean((np.asarray(out["arm_action"]) - gt[:, :6]) ** 2))
        hand_loss = float(np.mean((np.asarray(out["hand_action"]) - gt[:, 6:]) ** 2))
        hand_zero = float(np.mean((np.asarray(out_zero["hand_action"]) - gt[:, 6:]) ** 2))
        if "gt_hand_index_norm" in batch_np and "hand_index_norm" in out:
            hand_index = float(np.mean((np.asarray(out["hand_index_norm"]) - batch_np["gt_hand_index_norm"]) ** 2))
            hand_index_acc = float(np.mean(np.asarray(out["hand_index"]) == batch_np["gt_hand_index"]))
        else:
            hand_index = 0.0
            hand_index_acc = 0.0
        total = arm_loss + hand_loss
        batch_size_actual = int(gt.shape[0])
        sums["arm"] += arm_loss * batch_size_actual
        sums["hand_full"] += hand_loss * batch_size_actual
        sums["hand_no_corr"] += hand_zero * batch_size_actual
        sums["hand_index"] += hand_index * batch_size_actual
        sums["hand_index_acc"] += hand_index_acc * batch_size_actual
        sums["total"] += total * batch_size_actual
        sums["n"] += batch_size_actual

    return {
        "arm_mse": sums["arm"] / sums["n"],
        "hand_mse_full": sums["hand_full"] / sums["n"],
        "hand_mse_no_correction": sums["hand_no_corr"] / sums["n"],
        "hand_index_mse": sums["hand_index"] / sums["n"],
        "hand_index_acc": sums["hand_index_acc"] / sums["n"],
        "total_mse": sums["total"] / sums["n"],
    }


def rollout_ar(
    model,
    params,
    batch_stats,
    traj: dict,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    num_samples: int,
    seed: int,
    window_size: int,
    hand_prior_source: str,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 0,
    hand_history_std_floor: float = 1e-6,
):
    T = traj["T"]
    actions = traj["actions"].numpy()
    main_imgs = traj["imgs_main"].float().numpy() / 255.0
    extra_imgs = traj["imgs_extra"].float().numpy() / 255.0
    latent_dim = int(model.latent_dim)

    eval_step = make_eval_step(model)
    dropout_step = make_dropout_step(model)
    rng = jax.random.PRNGKey(seed)

    ar_runs = np.zeros((num_samples, T, 12), dtype=np.float32)
    arm_action_all = np.zeros((num_samples, T, 6), dtype=np.float32)
    hand_action_all = np.zeros((num_samples, T, 6), dtype=np.float32)
    hand_no_corr_all = np.zeros((num_samples, T, 6), dtype=np.float32)
    delta_z_all = np.zeros((num_samples, T, latent_dim), dtype=np.float32)
    mu_prior_all = np.zeros((num_samples, T, latent_dim), dtype=np.float32)
    z_ctrl_all = np.zeros((num_samples, T, latent_dim), dtype=np.float32)

    for s in range(num_samples):
        action_seq = actions.copy()
        for t in range(T):
            state = ((action_seq[t : t + 1] - action_mean) / action_std).astype(np.float32)
            raw_window = build_past_window_np(action_seq[:, 6:12], t, window_size)[None, ...].astype(np.float32)
            policy_window = policy_hand_window_from_raw(
                hand_prior_source,
                raw_window,
                action_mean,
                action_std,
                hand_history_std_floor,
            )
            img_main = main_imgs[t : t + 1].astype(np.float32)
            img_extra = extra_imgs[t : t + 1].astype(np.float32)
            rng, dropout_key = jax.random.split(rng)
            if enable_image_random_crop:
                rng, crop_key = jax.random.split(rng)
                img_main, img_extra = paired_edge_random_crop_nchw_np(
                    img_main,
                    img_extra,
                    crop_key,
                    random_crop_padding,
                )
            out = dropout_step(
                params,
                batch_stats,
                jnp.asarray(img_main),
                jnp.asarray(img_extra),
                jnp.asarray(state),
                jnp.asarray(policy_window),
                dropout_key,
            )
            pred = np.asarray(out["action_pred"][0])
            ar_runs[s, t, :] = pred
            arm_action_all[s, t, :] = np.asarray(out["arm_action"][0])
            hand_action_all[s, t, :] = np.asarray(out["hand_action"][0])
            hand_no_corr_all[s, t, :] = np.asarray(out["hand_no_corr"][0])
            delta_z_all[s, t, :] = np.asarray(out["delta_z"][0])
            mu_prior_all[s, t, :] = np.asarray(out["mu_prior"][0])
            z_ctrl_all[s, t, :] = np.asarray(out["z_ctrl"][0])
            if t + 1 < T:
                action_seq[t + 1, :] = pred

    no_corr = np.zeros((T, 12), dtype=np.float32)
    action_seq = actions.copy()
    for t in range(T):
        state = ((action_seq[t : t + 1] - action_mean) / action_std).astype(np.float32)
        raw_window = build_past_window_np(action_seq[:, 6:12], t, window_size)[None, ...].astype(np.float32)
        policy_window = policy_hand_window_from_raw(
            hand_prior_source,
            raw_window,
            action_mean,
            action_std,
            hand_history_std_floor,
        )
        img_main = main_imgs[t : t + 1].astype(np.float32)
        img_extra = extra_imgs[t : t + 1].astype(np.float32)
        if enable_image_random_crop:
            rng, crop_key = jax.random.split(rng)
            img_main, img_extra = paired_edge_random_crop_nchw_np(
                img_main,
                img_extra,
                crop_key,
                random_crop_padding,
            )
        out = eval_step(
            params,
            batch_stats,
            jnp.asarray(img_main),
            jnp.asarray(img_extra),
            jnp.asarray(state),
            jnp.asarray(policy_window),
            zero_delta=True,
        )
        pred = np.asarray(out["action_pred"][0])
        no_corr[t, :] = pred
        if t + 1 < T:
            action_seq[t + 1, :] = pred

    return {
        "ar_runs": ar_runs,
        "arm_action": arm_action_all,
        "hand_action": hand_action_all,
        "hand_no_corr": hand_no_corr_all,
        "no_corr": no_corr,
        "delta_z": delta_z_all,
        "mu_prior": mu_prior_all,
        "z_ctrl": z_ctrl_all,
    }


def _safe_name(value) -> str:
    return str(value).replace(os.sep, "_").replace(" ", "_")


def save_debug_trace(debug_dir: str, traj_id, gt, gt_target, rollout, onset_step, policy_family: str):
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"trajectory_{_safe_name(traj_id)}_debug.npz")
    np.savez_compressed(
        path,
        policy_family=np.asarray(policy_family),
        traj_id=np.asarray(str(traj_id)),
        onset_step=np.asarray(onset_step, dtype=np.int32),
        gt_action=np.asarray(gt, dtype=np.float32),
        gt_target=np.asarray(gt_target, dtype=np.float32),
        ar_action_pred=np.asarray(rollout["ar_runs"], dtype=np.float32),
        ar_arm_action=np.asarray(rollout["arm_action"], dtype=np.float32),
        ar_hand_action=np.asarray(rollout["hand_action"], dtype=np.float32),
        ar_hand_no_corr=np.asarray(rollout["hand_no_corr"], dtype=np.float32),
        no_corr_action_pred=np.asarray(rollout["no_corr"], dtype=np.float32),
        delta_z=np.asarray(rollout["delta_z"], dtype=np.float32),
        mu_prior=np.asarray(rollout["mu_prior"], dtype=np.float32),
        z_ctrl=np.asarray(rollout["z_ctrl"], dtype=np.float32),
    )
    dz_norm = np.linalg.norm(rollout["delta_z"].reshape(-1, rollout["delta_z"].shape[-1]), axis=-1)
    hand_mse = float(((rollout["hand_action"] - gt_target[None, :, 6:12]) ** 2).mean())
    return {
        "traj_id": str(traj_id),
        "path": path,
        "delta_z_norm_mean": float(dz_norm.mean()),
        "delta_z_norm_std": float(dz_norm.std()),
        "hand_action_mse": hand_mse,
    }


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, params, batch_stats, payload = load_policy_and_metadata(args.ckpt_dir)
    test_dir = args.test_dir or payload["train_args"].get("test_dir")
    if test_dir is None:
        test_dir = os.path.join(_PROJ_ROOT, "data/example_task/demos/success/test")

    action_mean = np.asarray(payload["action_mean"], dtype=np.float32)
    action_std = np.asarray(payload["action_std"], dtype=np.float32)
    window_size = int(payload["model_args"]["window_size"])
    hand_prior_source = payload["model_args"].get("hand_prior_source", "vae")
    random_crop_padding = resolve_random_crop_padding(args, payload)
    hand_history_std_floor = resolve_hand_history_std_floor(args, payload)
    print(
        "Eval enable_image_random_crop = "
        f"{args.enable_image_random_crop} (padding={random_crop_padding})"
    )
    print(f"Eval hand_history_std_floor = {hand_history_std_floor:g}")
    hand_codebook = None
    if hand_prior_source == "vq_codebook":
        if "hand_codebook" in payload:
            hand_codebook = np.asarray(payload["hand_codebook"], dtype=np.float32)
        else:
            hand_codebook = np.load(payload["model_args"]["hand_codebook_path"]).astype(np.float32)

    torch.manual_seed(args.seed)
    val_metrics = evaluate_validation_mse(
        model,
        params,
        batch_stats,
        test_dir,
        args.batch_size,
        action_mean,
        action_std,
        window_size,
        hand_prior_source,
        hand_codebook=hand_codebook,
        enable_image_random_crop=args.enable_image_random_crop,
        random_crop_padding=random_crop_padding,
        hand_history_std_floor=hand_history_std_floor,
    )

    files = sorted(glob.glob(os.path.join(test_dir, "trajectory_*_demo_expert.pt")))
    if args.max_trajs is not None:
        files = files[: args.max_trajs]
    print(f"Found {len(files)} test trajectories in {test_dir}")

    xtick_step = 5
    max_T_raw = 0
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        max_T_raw = max(max_T_raw, int(data["actions"].shape[0]))
    global_xlim = int(np.ceil(max_T_raw / xtick_step) * xtick_step) if files else 0

    all_results = []
    debug_files = []
    for fi, f in enumerate(files):
        traj = load_trajectory(f)
        traj_id = traj["traj_id"]
        T = traj["T"]
        gt = traj["actions"].numpy()
        gt_target = np.concatenate([gt[1:], gt[-1:]], axis=0)

        print(f"\n[{fi + 1}/{len(files)}] Trajectory {traj_id} (T={T})")
        rollout = rollout_ar(
            model=model,
            params=params,
            batch_stats=batch_stats,
            traj=traj,
            action_mean=action_mean,
            action_std=action_std,
            num_samples=args.num_samples,
            seed=args.seed + fi,
            window_size=window_size,
            hand_prior_source=hand_prior_source,
            enable_image_random_crop=args.enable_image_random_crop,
            random_crop_padding=random_crop_padding,
            hand_history_std_floor=hand_history_std_floor,
        )

        ar_runs = rollout["ar_runs"]
        no_corr = rollout["no_corr"]

        ar_arm_mse = float(((ar_runs[:, :, :6] - gt_target[:, :6]) ** 2).mean())
        ar_hand_mse = float(((ar_runs[:, :, 6:] - gt_target[:, 6:]) ** 2).mean())
        nc_arm_mse = float(((no_corr[:, :6] - gt_target[:, :6]) ** 2).mean())
        nc_hand_mse = float(((no_corr[:, 6:] - gt_target[:, 6:]) ** 2).mean())
        copy_arm_mse = float(((gt[:, :6] - gt_target[:, :6]) ** 2).mean())
        copy_hand_mse = float(((gt[:, 6:] - gt_target[:, 6:]) ** 2).mean())
        onset_step = detect_grasp_onset(gt)

        result = {
            "traj_id": traj_id,
            "T": T,
            "onset_step": onset_step,
            "ar_arm_mse": ar_arm_mse,
            "ar_hand_mse": ar_hand_mse,
            "nc_arm_mse": nc_arm_mse,
            "nc_hand_mse": nc_hand_mse,
            "copy_arm_mse": copy_arm_mse,
            "copy_hand_mse": copy_hand_mse,
            "delta_z": rollout["delta_z"],
            "mu_prior": rollout["mu_prior"],
            "z_ctrl": rollout["z_ctrl"],
        }
        all_results.append(result)
        if args.debug_dir is not None:
            debug_files.append(
                save_debug_trace(
                    args.debug_dir,
                    traj_id,
                    gt,
                    gt_target,
                    rollout,
                    onset_step,
                    policy_family="behavior_clone",
                )
            )

        print(f"  AR  arm={ar_arm_mse:.6f}  hand={ar_hand_mse:.6f}")
        print(f"  NC  arm={nc_arm_mse:.6f}  hand={nc_hand_mse:.6f}")

        plot_trajectory_actions(
            traj_id,
            T,
            gt_target,
            ar_runs,
            no_corr,
            args.num_samples,
            args.output_dir,
            onset_step=onset_step,
            xlim_max=global_xlim,
            xtick_step=xtick_step,
        )
        plot_trajectory_mse(
            traj_id,
            T,
            gt_target,
            ar_runs,
            no_corr,
            args.output_dir,
            onset_step=onset_step,
            xlim_max=global_xlim,
            xtick_step=xtick_step,
        )

    mean_ar_arm = float(np.mean([r["ar_arm_mse"] for r in all_results])) if all_results else 0.0
    mean_ar_hand = float(np.mean([r["ar_hand_mse"] for r in all_results])) if all_results else 0.0
    mean_nc_arm = float(np.mean([r["nc_arm_mse"] for r in all_results])) if all_results else 0.0
    mean_nc_hand = float(np.mean([r["nc_hand_mse"] for r in all_results])) if all_results else 0.0
    mean_copy_arm = float(np.mean([r["copy_arm_mse"] for r in all_results])) if all_results else 0.0
    mean_copy_hand = float(np.mean([r["copy_hand_mse"] for r in all_results])) if all_results else 0.0

    all_dz = (
        np.concatenate([r["delta_z"].reshape(-1, r["delta_z"].shape[-1]) for r in all_results], axis=0)
        if all_results
        else np.zeros((1, int(model.latent_dim)), dtype=np.float32)
    )
    dz_norm = np.linalg.norm(all_dz, axis=-1)

    summary = {
        "backend": "jax",
        "ckpt_dir": args.ckpt_dir,
        "num_samples": args.num_samples,
        "num_trajectories": len(all_results),
        "enable_image_random_crop": bool(args.enable_image_random_crop),
        "random_crop_padding": int(random_crop_padding),
        "val_arm_mse": float(val_metrics["arm_mse"]),
        "val_hand_mse_full": float(val_metrics["hand_mse_full"]),
        "val_hand_mse_no_correction": float(val_metrics["hand_mse_no_correction"]),
        "val_hand_index_mse": float(val_metrics.get("hand_index_mse", 0.0)),
        "val_hand_index_acc": float(val_metrics.get("hand_index_acc", 0.0)),
        "val_total_mse": float(val_metrics["total_mse"]),
        "vision_gain_val": float(val_metrics["hand_mse_no_correction"] - val_metrics["hand_mse_full"]),
        "ar_arm_mse": mean_ar_arm,
        "ar_hand_mse": mean_ar_hand,
        "nc_arm_mse": mean_nc_arm,
        "nc_hand_mse": mean_nc_hand,
        "copy_arm_mse": mean_copy_arm,
        "copy_hand_mse": mean_copy_hand,
        "vision_gain_ar": float(mean_nc_hand - mean_ar_hand),
        "delta_z_norm_mean": float(dz_norm.mean()),
        "delta_z_norm_std": float(dz_norm.std()),
        "debug_dir": args.debug_dir,
        "debug_files": debug_files,
        "per_trajectory": [
            {
                "traj_id": r["traj_id"],
                "T": r["T"],
                "onset_step": r["onset_step"],
                "ar_arm_mse": r["ar_arm_mse"],
                "ar_hand_mse": r["ar_hand_mse"],
                "nc_arm_mse": r["nc_arm_mse"],
                "nc_hand_mse": r["nc_hand_mse"],
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_summary(all_results, args.output_dir, args.num_samples)
    plot_per_trajectory_bar(all_results, args.output_dir)
    plot_latent_diagnostics(all_results, args.output_dir, args.num_samples)
    print(f"\nSummary JSON: {summary_path}")
    print(f"All figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

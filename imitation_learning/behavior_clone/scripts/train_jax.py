"""Train integrated single-step behavior_clone policy with Flax + Optax."""

import argparse
import copy
import importlib.util
import json
import os
import pickle
import time
from typing import Any, Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import train_state
from torch.utils.data import DataLoader


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_VAE_ROOT = os.path.join(_PROJ_ROOT, "vae")


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dataset_mod = _load(os.path.join(_BC_ROOT, "model/bc_dataset.py"), "bc_dataset_behavior_clone_jax")
_policy_mod = _load(os.path.join(_BC_ROOT, "model/bc_policy_jax.py"), "bc_policy_behavior_clone_jax")
_compat_mod = _load(os.path.join(_BC_ROOT, "model/checkpoint_compat.py"), "bc_ckpt_compat_behavior_clone_jax")
_utils_mod = _load(os.path.join(_VAE_ROOT, "model/utils.py"), "vae_utils_behavior_clone_jax")

BCDataset = _dataset_mod.BCDataset
compute_action_stats = _dataset_mod.compute_action_stats
BCPolicy = _policy_mod.BCPolicy
count_params = _policy_mod.count_params
DEFAULT_HIL_SERL_ROOT = getattr(_policy_mod, "DEFAULT_HIL_SERL_ROOT", "")
load_resnet18_flax_variables = _compat_mod.load_resnet18_flax_variables
load_resnet18_config = _compat_mod.load_resnet18_config
resolve_jax_checkpoint_dir = _compat_mod.resolve_jax_checkpoint_dir
restore_jax_checkpoint = _compat_mod.restore_jax_checkpoint
save_jax_checkpoint_payload = _compat_mod.save_jax_checkpoint
cosine_scheduler = _utils_mod.cosine_scheduler


class TrainState(train_state.TrainState):
    batch_stats: Any


def get_args():
    p = argparse.ArgumentParser(description="Train integrated behavior_clone policy with JAX")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--vae_ckpt", type=str, default="pretrained_models/jax_ckpt/hand_vae")
    p.add_argument("--vae_latent_dim", type=int, default=2, help="Used only when --vae_ckpt=none.")
    p.add_argument("--vae_hidden_dim", type=int, default=256, help="Used only when --vae_ckpt=none.")
    p.add_argument(
        "--vae_encoder_type",
        type=str,
        default="mlp",
        choices=["mlp", "causal_conv"],
        help="Used only when --vae_ckpt=none.",
    )
    p.add_argument("--vae_num_hidden_layers", type=int, default=1, help="Used only when --vae_ckpt=none.")
    p.add_argument("--vae_beta", type=float, default=0.001, help="Used only when --vae_ckpt=none.")
    p.add_argument("--vae_recon_aux_weight", type=float, default=0.0, help="Used only when --vae_ckpt=none.")
    p.add_argument("--vae_free_bits", type=float, default=0.0, help="Used only when --vae_ckpt=none.")
    p.add_argument(
        "--hand_prior_source",
        type=str,
        default="vae",
        choices=["vae", "mlp_direct", "decoder_only", "vq_codebook", "pca_raw"],
        help=(
            "Use frozen VAE hand prior, direct 6D MLP hand head, "
            "direct 2D decoder-only latent head, DQ-RISE style VQ codebook index, "
            "or raw hand-history PCA latent head."
        ),
    )
    p.add_argument("--hand_pca_model", type=str, default=None, help="pca_model.npz for hand_prior_source=pca_raw")
    p.add_argument(
        "--backbone_impl",
        type=str,
        default="hf_resnet18",
        choices=["hf_resnet18", "hil_serl_resnet10"],
    )
    p.add_argument("--resnet_path", type=str, default="microsoft/resnet-18")
    p.add_argument("--hil_serl_root", type=str, default=DEFAULT_HIL_SERL_ROOT)
    p.add_argument(
        "--resnet10_ckpt",
        type=str,
        default="pretrained_models/resnet10_params.pkl",
    )
    p.add_argument("--output_dir", type=str, default="outputs/behavior_clone_jax")
    p.add_argument("--hand_codebook", type=str, default=None, help="sorted_codebook.npy for hand_prior_source=vq_codebook")
    p.add_argument("--hand_index_weight", type=float, default=1.0)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument(
        "--bc_image_keys",
        type=str,
        default="global,wrist",
        help="Comma or space separated image keys for BC policy, e.g. global,wrist,global_2.",
    )
    p.add_argument("--state_encoder", type=str, default="mlp", choices=["mlp", "linear64", "raw"])
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Optional JAX BC checkpoint dir/file to resume from. Continues until --total_steps.",
    )
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--noise_std_hand", type=float, default=0.1)
    p.add_argument("--noise_std_arm", type=float, default=0.1)
    p.add_argument(
        "--hand_history_std_floor",
        type=float,
        default=0.02,
        help=(
            "Minimum std used only when normalizing past_hand_state_win for "
            "direct hand-history heads. action_std itself is still saved unchanged."
        ),
    )
    p.add_argument(
        "--enable_image_random_crop",
        action="store_true",
        help="Enable paired edge-padded random crop on training images. Disabled by default.",
    )
    p.add_argument(
        "--random_crop_padding",
        type=int,
        default=4,
        help="Image random crop padding used only when --enable_image_random_crop is set.",
    )
    p.add_argument("--arm_weight", type=float, default=1.0)
    p.add_argument("--hand_weight", type=float, default=1.0)
    p.add_argument("--hand_loss_space", type=str, default="action", choices=["action"])
    p.add_argument("--reg_drift", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--eval_freq", type=int, default=1000)
    p.add_argument("--save_freq", type=int, default=5000)
    return p.parse_args()


def torch_batch_to_numpy(batch):
    return {k: np.asarray(v.numpy(), dtype=np.float32) for k, v in batch.items()}


def seed_torch_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def parse_image_keys(value: str) -> tuple[str, ...]:
    keys = tuple(key.strip() for key in str(value).replace(",", " ").split() if key.strip())
    if len(keys) < 2:
        raise ValueError(f"--bc_image_keys requires at least two keys, got {value!r}")
    return keys


def extra_images_from_batch(batch_np: dict[str, np.ndarray]) -> tuple[jax.Array, ...]:
    extra = batch_np.get("extra_images")
    if extra is None:
        return ()
    extra = np.asarray(extra, dtype=np.float32)
    if extra.ndim != 5:
        raise ValueError(f"extra_images must have shape (B, N, C, H, W), got {extra.shape}")
    return tuple(jnp.asarray(extra[:, idx]) for idx in range(extra.shape[1]))


def policy_hand_window_from_batch(batch_np, hand_prior_source: str):
    if hand_prior_source in {"mlp_direct", "decoder_only", "vq_codebook"}:
        return batch_np["past_hand_state_win"]
    if hand_prior_source == "pca_raw":
        return batch_np["past_hand_win_raw"]
    return batch_np["past_hand_win"]


def load_hand_codebook(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    codebook = np.load(path).astype(np.float32)
    if codebook.ndim != 2 or codebook.shape[1] != 6:
        raise ValueError(f"--hand_codebook must point to an array of shape (N, 6), got {codebook.shape}")
    return codebook


def load_hand_pca_model(path: str | None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if path is None:
        return None, None
    payload = np.load(path)
    if "mean" not in payload or "components" not in payload:
        raise ValueError(f"--hand_pca_model must contain 'mean' and 'components': {path}")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    components = np.asarray(payload["components"], dtype=np.float32)
    if mean.shape != (6,):
        raise ValueError(f"--hand_pca_model mean must have shape (6,), got {mean.shape}")
    if components.ndim != 2 or components.shape[0] < 2 or components.shape[1] != 6:
        raise ValueError(
            f"--hand_pca_model components must have shape at least (2, 6), got {components.shape}"
        )
    return mean, components


def is_none_vae_ckpt(path: str | None) -> bool:
    return path is None or str(path).strip().lower() in {"", "none", "null"}


def build_random_vae_model_args(args) -> Dict[str, Any]:
    return {
        "action_dim": 6,
        "window_size": args.window_size,
        "hidden_dim": args.vae_hidden_dim,
        "latent_dim": args.vae_latent_dim,
        "beta": args.vae_beta,
        "encoder_type": args.vae_encoder_type,
        "num_hidden_layers": args.vae_num_hidden_layers,
        "recon_aux_weight": args.vae_recon_aux_weight,
        "free_bits": args.vae_free_bits,
    }


def build_model_args(args, vae_model_args=None, hand_codebook=None, hand_pca_mean=None, hand_pca_components=None):
    bc_image_keys = parse_image_keys(args.bc_image_keys)
    if args.hand_prior_source == "vq_codebook":
        prior_cfg = {"latent_dim": 1}
    elif args.hand_prior_source == "mlp_direct":
        prior_cfg = None
    elif args.hand_prior_source == "pca_raw":
        if hand_pca_mean is None or hand_pca_components is None:
            raise ValueError("--hand_pca_model is required when --hand_prior_source=pca_raw")
        prior_cfg = None
    elif args.hand_prior_source in {"vae", "decoder_only"}:
        if vae_model_args is None:
            raise ValueError(
                f"vae_model_args must be provided when hand_prior_source={args.hand_prior_source!r}"
            )
        if args.hand_prior_source == "decoder_only" and int(vae_model_args["latent_dim"]) != 2:
            raise ValueError(f"decoder_only requires a 2D VAE latent, got {vae_model_args['latent_dim']}")
        prior_cfg = dict(vae_model_args)
    else:
        raise ValueError(f"Unknown hand_prior_source: {args.hand_prior_source}")
    model_args = {
        "state_encoder": args.state_encoder,
        "freeze_backbone": args.freeze_backbone,
        "backbone_lr_scale": args.backbone_lr_scale,
        "dropout": args.dropout,
        "window_size": args.window_size,
        "arm_state_dim": 6,
        "hand_state_dim": 6,
        "hand_prior_source": args.hand_prior_source,
        "hand_history_std_floor": float(args.hand_history_std_floor),
        "bc_image_keys": bc_image_keys,
        "image_keys": bc_image_keys,
        "num_image_views": len(bc_image_keys),
        "freeze_vae": not (args.hand_prior_source == "vae" and is_none_vae_ckpt(args.vae_ckpt)),
        "vae_init": (
            "random_trainable"
            if args.hand_prior_source == "vae" and is_none_vae_ckpt(args.vae_ckpt)
            else ("pretrained_frozen" if args.hand_prior_source in {"vae", "decoder_only"} else None)
        ),
    }
    if prior_cfg is not None:
        model_args["vae"] = prior_cfg
    if args.hand_prior_source == "mlp_direct":
        model_args["direct_head_type"] = "core_action_encoded_normalized_hand_history_direct_12d"
        model_args["requested_state_encoder"] = args.state_encoder
        model_args["direct_hand_input"] = "encoded_normalized_past_hand_state_win"
    if args.hand_prior_source == "decoder_only":
        model_args["direct_head_type"] = "core_action_encoded_normalized_hand_history_decoder_only_8d"
        model_args["requested_state_encoder"] = args.state_encoder
        model_args["direct_hand_input"] = "encoded_normalized_past_hand_state_win"
    if args.hand_prior_source == "vq_codebook":
        if hand_codebook is None:
            raise ValueError("--hand_codebook is required when --hand_prior_source=vq_codebook")
        model_args.update(
            {
                "direct_head_type": "core_action_encoded_normalized_hand_history_vq_codebook_7d",
                "requested_state_encoder": args.state_encoder,
                "direct_hand_input": "encoded_normalized_past_hand_state_win",
                "hand_codebook_path": args.hand_codebook,
                "hand_codebook_num_codes": int(hand_codebook.shape[0]),
                "hand_index_weight": float(args.hand_index_weight),
            }
        )
    if args.hand_prior_source == "pca_raw":
        model_args.update(
            {
                "direct_head_type": "core_action_encoded_raw_pca_history_pca_raw_8d",
                "requested_state_encoder": args.state_encoder,
                "direct_hand_input": "encoded_raw_past_hand_win_raw_pca2",
                "hand_pca_dim": 2,
                "hand_pca_mean_shape": list(np.asarray(hand_pca_mean).shape),
                "hand_pca_components_shape": list(np.asarray(hand_pca_components).shape),
            }
        )
    if args.backbone_impl == "hf_resnet18":
        model_args.update(
            {
                "backbone_type": "hf_resnet18",
                "backbone_config": load_resnet18_config(args.resnet_path),
                "resnet_path": args.resnet_path,
            }
        )
    elif args.backbone_impl == "hil_serl_resnet10":
        model_args.update(
            {
                "backbone_type": "hil_serl_resnet10",
                "backbone_config": None,
                "hil_serl_root": args.hil_serl_root,
                "resnet10_ckpt": args.resnet10_ckpt,
                "hil_serl_pooling_method": "spatial_learned_embeddings",
                "hil_serl_num_spatial_blocks": 8,
                "hil_serl_bottleneck_dim": 256,
            }
        )
    else:
        raise ValueError(f"Unknown backbone_impl: {args.backbone_impl}")
    return model_args


def _label_subtree(tree, label):
    return jax.tree_util.tree_map(lambda _: label, tree)


def build_param_labels(params, freeze_backbone: bool, backbone_type: str, train_vae: bool = False):
    labels = {}
    for name, subtree in flax.core.unfreeze(params).items():
        if name == "vae":
            labels[name] = _label_subtree(subtree, "head" if train_vae else "frozen")
        elif name == "backbone":
            if backbone_type == "hil_serl_resnet10":
                backbone_labels = _label_subtree(subtree, "head")
                if freeze_backbone:
                    encoder_keys = [
                        key for key in subtree.keys()
                        if key.startswith("encoder_img_")
                    ]
                    for encoder_key in encoder_keys:
                        if encoder_key in backbone_labels and "pretrained_encoder" in backbone_labels[encoder_key]:
                            backbone_labels[encoder_key]["pretrained_encoder"] = _label_subtree(
                                subtree[encoder_key]["pretrained_encoder"],
                                "frozen",
                            )
                labels[name] = backbone_labels
            else:
                label = "frozen" if freeze_backbone else "backbone"
                labels[name] = _label_subtree(subtree, label)
        else:
            labels[name] = _label_subtree(subtree, "head")
    return flax.core.freeze(labels)


def load_hil_serl_resnet10_params(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HiL-SERL ResNet-10 checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    if hasattr(params, "unfreeze"):
        params = params.unfreeze()
    return params


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
        f"JAX Behavior Clone  (reg_drift={args.reg_drift}, hand_noise={args.noise_std_hand}, "
        f"arm_noise={args.noise_std_arm}, steps={args.total_steps})",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(steps, history["train_total"], color="blue", alpha=0.4, linewidth=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train Total Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, history["train_arm"], color="green", linewidth=1.0, label="arm")
    ax.plot(steps, history["train_hand"], color="red", linewidth=1.0, label="hand")
    ax.plot(steps, history["train_drift"], color="purple", linewidth=1.0, label="drift")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Train Arm / Hand / Drift")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    if eval_steps:
        gains = [nc - full for full, nc in zip(history["val_hand_no_corr"], history["val_hand_full"])]
        ax.plot(eval_steps, history["val_arm"], "go-", markersize=4, label="val arm")
        ax.plot(eval_steps, history["val_hand_full"], "ro-", markersize=4, label="val hand")
        ax.plot(eval_steps, history["val_hand_no_corr"], "k--", linewidth=1.0, label="val no-corr")
        ax.plot(eval_steps, gains, "mo-", markersize=4, label="vision gain")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Validation Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, history["val_total"], "bo-", markersize=4, label="val total")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Total MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, history["train_lr"], color="teal", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine LR")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if eval_steps:
        gains = [nc - full for full, nc in zip(history["val_hand_no_corr"], history["val_hand_full"])]
        ax.plot(eval_steps, gains, "mo-", markersize=4)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("no_corr - hand")
    ax.set_title("Vision Gain")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves_jax.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {fig_path}")


def save_summary(history, output_dir, args, model_args, final_metrics, action_mean, action_std, vae_backend):
    summary = {
        "backend": "jax",
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "output_dir": args.output_dir,
        "vae_ckpt_init": args.vae_ckpt,
        "vae_backend": vae_backend,
        "action_mean": [float(x) for x in np.asarray(action_mean).tolist()],
        "action_std": [float(x) for x in np.asarray(action_std).tolist()],
        "model_args": model_args,
        "train_args": vars(args),
        "final_metrics": final_metrics,
        "history": history,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_vae_init(vae_ckpt_root: str):
    ckpt_dir = resolve_jax_checkpoint_dir(vae_ckpt_root)
    payload = restore_jax_checkpoint(ckpt_dir)
    return payload["params"], payload["model_args"], payload.get("backend", "jax"), ckpt_dir


def load_vae_metadata(vae_ckpt_root: str):
    ckpt_dir = resolve_jax_checkpoint_dir(vae_ckpt_root)
    payload = restore_jax_checkpoint(ckpt_dir)
    return payload["model_args"], payload.get("backend", "jax"), ckpt_dir


def load_resume_payload(resume_ckpt: str) -> Dict[str, Any]:
    if resume_ckpt.endswith(".pkl"):
        with open(resume_ckpt, "rb") as f:
            return pickle.load(f)
    ckpt_dir = resolve_jax_checkpoint_dir(resume_ckpt)
    return restore_jax_checkpoint(ckpt_dir)


def create_train_state(
    args,
    vae_params,
    vae_model_args,
    lr_schedule,
    hand_codebook=None,
    hand_pca_mean=None,
    hand_pca_components=None,
):
    model_args = build_model_args(
        args,
        vae_model_args,
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
    )
    model = BCPolicy(
        vae_model_args=model_args.get("vae"),
        hand_prior_source=model_args.get("hand_prior_source", "vae"),
        freeze_vae=model_args.get("freeze_vae", True),
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
        hand_pca_dim=model_args.get("hand_pca_dim", 2),
        window_size=model_args["window_size"],
        backbone_type=model_args.get("backbone_type", "hf_resnet18"),
        backbone_config=model_args["backbone_config"],
        state_encoder_type=model_args["state_encoder"],
        arm_state_dim=model_args["arm_state_dim"],
        hand_state_dim=model_args.get("hand_state_dim", 6),
        dropout=model_args["dropout"],
        hil_serl_root=model_args.get("hil_serl_root", DEFAULT_HIL_SERL_ROOT),
        hil_serl_pooling_method=model_args.get("hil_serl_pooling_method", "spatial_learned_embeddings"),
        hil_serl_num_spatial_blocks=model_args.get("hil_serl_num_spatial_blocks", 8),
        hil_serl_bottleneck_dim=model_args.get("hil_serl_bottleneck_dim", 256),
        num_image_views=int(model_args.get("num_image_views", 2)),
    )

    init_key = jax.random.PRNGKey(args.seed)
    dummy_img = jnp.zeros((1, 3, 128, 128), dtype=jnp.float32)
    dummy_extra_images = tuple(
        dummy_img for _ in range(max(0, int(model_args.get("num_image_views", 2)) - 2))
    )
    dummy_state = jnp.zeros((1, 12), dtype=jnp.float32)
    dummy_win = jnp.zeros((1, args.window_size, 6), dtype=jnp.float32)
    variables = model.init(
        {"params": init_key, "dropout": init_key},
        dummy_img,
        dummy_img,
        dummy_state,
        dummy_win,
        extra_images=dummy_extra_images,
        deterministic=False,
        train_backbone=False,
    )

    params = flax.core.unfreeze(variables["params"])
    batch_stats = flax.core.unfreeze(variables.get("batch_stats", {}))
    if model_args["hand_prior_source"] in {"vae", "decoder_only"} and vae_params is not None:
        params["vae"] = flax.core.unfreeze(vae_params)

    if model_args["backbone_type"] == "hf_resnet18":
        backbone_params, backbone_batch_stats = load_resnet18_flax_variables(args.resnet_path)
        params["backbone"]["resnet"] = backbone_params
        batch_stats["backbone"]["resnet"] = backbone_batch_stats
    elif model_args["backbone_type"] == "hil_serl_resnet10":
        pretrained_params = load_hil_serl_resnet10_params(args.resnet10_ckpt)
        params["backbone"]["encoder_img_main"]["pretrained_encoder"] = copy.deepcopy(pretrained_params)
        params["backbone"]["encoder_img_extra"]["pretrained_encoder"] = copy.deepcopy(pretrained_params)
        for view_idx in range(2, int(model_args.get("num_image_views", 2))):
            encoder_key = f"encoder_img_extra_{view_idx}"
            params["backbone"][encoder_key]["pretrained_encoder"] = copy.deepcopy(pretrained_params)
    else:
        raise ValueError(f"Unknown backbone_type: {model_args['backbone_type']}")

    params = flax.core.freeze(params)
    batch_stats = flax.core.freeze(batch_stats)

    lr_schedule_jax = jnp.asarray(lr_schedule, dtype=jnp.float32)

    def head_schedule_fn(step):
        step = jnp.minimum(step, lr_schedule_jax.shape[0] - 1)
        return lr_schedule_jax[step]

    def backbone_schedule_fn(step):
        step = jnp.minimum(step, lr_schedule_jax.shape[0] - 1)
        return args.backbone_lr_scale * lr_schedule_jax[step]

    labels = build_param_labels(
        params,
        args.freeze_backbone,
        model_args["backbone_type"],
        train_vae=not model_args.get("freeze_vae", True),
    )
    tx = optax.multi_transform(
        {
            "head": optax.chain(
                optax.clip_by_global_norm(args.clip_grad),
                optax.adamw(learning_rate=head_schedule_fn, weight_decay=args.weight_decay),
            ),
            "backbone": optax.chain(
                optax.clip_by_global_norm(args.clip_grad),
                optax.adamw(learning_rate=backbone_schedule_fn, weight_decay=args.weight_decay),
            ),
            "frozen": optax.set_to_zero(),
        },
        labels,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    has_batch_stats = bool(flax.traverse_util.flatten_dict(flax.core.unfreeze(batch_stats)))
    return model, state, model_args, has_batch_stats


def make_train_step(
    model,
    arm_weight: float,
    hand_weight: float,
    hand_loss_space: str,
    reg_drift: float,
    freeze_backbone: bool,
    has_batch_stats: bool,
    hand_prior_source: str,
    hand_index_weight: float,
):
    @jax.jit
    def train_step(state, img_main, img_extra, extra_images, state_in, past_hand_win, gt_action, gt_delta_z, gt_hand_index_norm, dropout_key):
        def loss_fn(params):
            variables = {"params": params}
            if has_batch_stats:
                variables["batch_stats"] = state.batch_stats
            apply_kwargs = dict(
                img_main=img_main,
                img_extra=img_extra,
                state=state_in,
                past_hand_win=past_hand_win,
                extra_images=extra_images,
                deterministic=False,
                train_backbone=not freeze_backbone,
            )
            if freeze_backbone or not has_batch_stats:
                out = model.apply(variables, **apply_kwargs, rngs={"dropout": dropout_key})
                new_batch_stats = state.batch_stats
            else:
                out, updates = model.apply(
                    variables,
                    **apply_kwargs,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                )
                new_batch_stats = updates["batch_stats"]

            arm_loss = jnp.mean(jnp.square(out["arm_action"] - gt_action[:, :6]))
            if hand_prior_source == "vq_codebook":
                hand_loss = jnp.mean(jnp.square(out["hand_index_norm"] - gt_hand_index_norm))
                drift_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                decoded_hand_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
                hand_latent_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                weighted_hand_loss = hand_index_weight * hand_loss
                total_loss = arm_weight * arm_loss + weighted_hand_loss
            elif hand_prior_source == "vae":
                hand_action_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
                if hand_loss_space == "latent":
                    hand_latent_loss = jnp.mean(jnp.square(out["delta_z"] - gt_delta_z))
                    hand_loss = hand_latent_loss
                else:
                    hand_latent_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                    hand_loss = hand_action_loss
                drift_loss = jnp.mean(jnp.square(out["hand_action"] - out["hand_no_corr"]))
                decoded_hand_loss = hand_action_loss
                weighted_hand_loss = hand_weight * hand_loss
                total_loss = arm_weight * arm_loss + weighted_hand_loss + reg_drift * drift_loss
            elif hand_prior_source in {"decoder_only", "pca_raw"}:
                hand_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
                drift_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                decoded_hand_loss = hand_loss
                hand_latent_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                weighted_hand_loss = hand_weight * hand_loss
                total_loss = arm_weight * arm_loss + weighted_hand_loss
            else:
                hand_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
                drift_loss = jnp.mean(jnp.square(out["hand_action"] - out["hand_no_corr"]))
                decoded_hand_loss = hand_loss
                hand_latent_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
                weighted_hand_loss = hand_weight * hand_loss
                total_loss = arm_weight * arm_loss + weighted_hand_loss + reg_drift * drift_loss
            metrics = {
                "total": total_loss,
                "arm": arm_loss,
                "hand": hand_loss,
                "hand_action": decoded_hand_loss,
                "hand_latent": hand_latent_loss,
                "weighted_arm": arm_weight * arm_loss,
                "weighted_hand": weighted_hand_loss,
                "drift": drift_loss,
                "weighted_drift": reg_drift * drift_loss,
                "decoded_hand": decoded_hand_loss,
            }
            return total_loss, (metrics, new_batch_stats)

        (loss, (metrics, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
        return state, metrics

    return train_step


def make_eval_step(model, has_batch_stats: bool, hand_prior_source: str):
    @jax.jit
    def eval_step(params, batch_stats, img_main, img_extra, extra_images, state_in, past_hand_win, gt_action, gt_hand_index_norm, gt_hand_index):
        variables = {"params": params}
        if has_batch_stats:
            variables["batch_stats"] = batch_stats
        out = model.apply(
            variables,
            img_main,
            img_extra,
            state_in,
            past_hand_win,
            extra_images=extra_images,
            deterministic=True,
            train_backbone=False,
        )
        out_zero = model.apply(
            variables,
            img_main,
            img_extra,
            state_in,
            past_hand_win,
            extra_images=extra_images,
            zero_delta=True,
            deterministic=True,
            train_backbone=False,
        )
        arm_loss = jnp.mean(jnp.square(out["arm_action"] - gt_action[:, :6]))
        if hand_prior_source == "vq_codebook":
            hand_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
            hand_zero = hand_loss
            hand_index_loss = jnp.mean(jnp.square(out["hand_index_norm"] - gt_hand_index_norm))
            pred_idx = out["hand_index"].astype(jnp.float32)
            hand_index_acc = jnp.mean((pred_idx == gt_hand_index).astype(jnp.float32))
        else:
            hand_loss = jnp.mean(jnp.square(out["hand_action"] - gt_action[:, 6:]))
            hand_zero = jnp.mean(jnp.square(out_zero["hand_action"] - gt_action[:, 6:]))
            hand_index_loss = jnp.asarray(0.0, dtype=arm_loss.dtype)
            hand_index_acc = jnp.asarray(0.0, dtype=arm_loss.dtype)
        total = arm_loss + hand_loss
        return {
            "arm": arm_loss,
            "hand": hand_loss,
            "hand_no_corr": hand_zero,
            "hand_index": hand_index_loss,
            "hand_index_acc": hand_index_acc,
            "total": total,
        }

    return eval_step


def evaluate_model(state, eval_step, loader, hand_prior_source: str):
    sums = {
        "arm": 0.0,
        "hand_full": 0.0,
        "hand_no_corr": 0.0,
        "hand_index": 0.0,
        "hand_index_acc": 0.0,
        "total": 0.0,
        "n": 0,
    }
    for batch in loader:
        batch_np = torch_batch_to_numpy(batch)
        gt_hand_index_norm = batch_np.get(
            "gt_hand_index_norm",
            np.zeros((batch_np["gt_action"].shape[0], 1), dtype=np.float32),
        )
        gt_hand_index = batch_np.get(
            "gt_hand_index",
            np.zeros((batch_np["gt_action"].shape[0],), dtype=np.float32),
        )
        metrics = eval_step(
            state.params,
            state.batch_stats,
            jnp.asarray(batch_np["img_main"]),
            jnp.asarray(batch_np["img_extra"]),
            extra_images_from_batch(batch_np),
            jnp.asarray(batch_np["state"]),
            jnp.asarray(policy_hand_window_from_batch(batch_np, hand_prior_source)),
            jnp.asarray(batch_np["gt_action"]),
            jnp.asarray(gt_hand_index_norm),
            jnp.asarray(gt_hand_index),
        )
        batch_size = int(batch_np["gt_action"].shape[0])
        sums["arm"] += float(metrics["arm"]) * batch_size
        sums["hand_full"] += float(metrics["hand"]) * batch_size
        sums["hand_no_corr"] += float(metrics["hand_no_corr"]) * batch_size
        sums["hand_index"] += float(metrics["hand_index"]) * batch_size
        sums["hand_index_acc"] += float(metrics["hand_index_acc"]) * batch_size
        sums["total"] += float(metrics["total"]) * batch_size
        sums["n"] += batch_size

    return {
        "arm_mse": sums["arm"] / sums["n"],
        "hand_mse_full": sums["hand_full"] / sums["n"],
        "hand_mse_no_correction": sums["hand_no_corr"] / sums["n"],
        "hand_index_mse": sums["hand_index"] / sums["n"],
        "hand_index_acc": sums["hand_index_acc"] / sums["n"],
        "total_mse": sums["total"] / sums["n"],
    }


def save_checkpoint(
    state,
    rng,
    step,
    output_dir,
    model_args,
    args,
    action_mean,
    action_std,
    vae_backend,
    hand_codebook=None,
    hand_pca_mean=None,
    hand_pca_components=None,
):
    ckpt_dir = os.path.join(output_dir, "jax_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "opt_state": state.opt_state,
        "step": int(step),
        "rng": np.asarray(rng),
        "model_args": model_args,
        "train_args": vars(args),
        "action_mean": np.asarray(action_mean, dtype=np.float32),
        "action_std": np.asarray(action_std, dtype=np.float32),
        "backend": "jax",
        "vae_backend": vae_backend,
        "vae_ckpt_init": args.vae_ckpt,
    }
    if hand_codebook is not None:
        payload["hand_codebook"] = np.asarray(hand_codebook, dtype=np.float32)
    if hand_pca_mean is not None and hand_pca_components is not None:
        payload["hand_pca_mean"] = np.asarray(hand_pca_mean, dtype=np.float32)
        payload["hand_pca_components"] = np.asarray(hand_pca_components, dtype=np.float32)
    return save_jax_checkpoint_payload(ckpt_dir, payload, step)


def main():
    args = get_args()
    if args.random_crop_padding < 0:
        raise ValueError(f"--random_crop_padding must be >= 0, got {args.random_crop_padding}")
    if args.hand_history_std_floor < 0:
        raise ValueError(f"--hand_history_std_floor must be >= 0, got {args.hand_history_std_floor}")
    bc_image_keys = parse_image_keys(args.bc_image_keys)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Computing action mean/std from {args.train_dir} ...")
    action_mean_t, action_std_t = compute_action_stats(args.train_dir)
    action_mean = np.asarray(action_mean_t.numpy(), dtype=np.float32)
    action_std = np.asarray(action_std_t.numpy(), dtype=np.float32)
    print(f"  action_mean = {[round(x, 3) for x in action_mean.tolist()]}")
    print(f"  action_std  = {[round(x, 3) for x in action_std.tolist()]}")
    print(f"  bc_image_keys = {bc_image_keys}")
    print(
        "  enable_image_random_crop = "
        f"{args.enable_image_random_crop} (padding={args.random_crop_padding}, train split only)"
    )
    print(f"  hand_history_std_floor = {args.hand_history_std_floor:g} (direct hand-history inputs only)")
    hand_codebook = load_hand_codebook(args.hand_codebook) if args.hand_prior_source == "vq_codebook" else None
    if hand_codebook is not None:
        print(f"  hand_codebook = {args.hand_codebook} shape={hand_codebook.shape}")
    hand_pca_mean, hand_pca_components = (
        load_hand_pca_model(args.hand_pca_model) if args.hand_prior_source == "pca_raw" else (None, None)
    )
    if hand_pca_mean is not None and hand_pca_components is not None:
        print(
            f"  hand_pca_model = {args.hand_pca_model} "
            f"mean={hand_pca_mean.shape} components={hand_pca_components.shape} dim=2"
        )

    train_ds = BCDataset(
        args.train_dir,
        action_mean=action_mean_t,
        action_std=action_std_t,
        window_size=args.window_size,
        noise_std_hand=args.noise_std_hand,
        noise_std_arm=args.noise_std_arm,
        hand_codebook=hand_codebook,
        image_keys=bc_image_keys,
        enable_image_random_crop=args.enable_image_random_crop,
        random_crop_padding=args.random_crop_padding,
        hand_history_std_floor=args.hand_history_std_floor,
    )
    test_ds = BCDataset(
        args.test_dir,
        action_mean=action_mean_t,
        action_std=action_std_t,
        window_size=args.window_size,
        noise_std_hand=0.0,
        noise_std_arm=0.0,
        hand_codebook=hand_codebook,
        image_keys=bc_image_keys,
        enable_image_random_crop=False,
        random_crop_padding=args.random_crop_padding,
        hand_history_std_floor=args.hand_history_std_floor,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=seed_torch_worker,
        generator=train_loader_generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if args.hand_prior_source == "vae":
        if is_none_vae_ckpt(args.vae_ckpt):
            vae_params = None
            vae_model_args = build_random_vae_model_args(args)
            vae_backend = "random_trainable"
            print(f"  vae_init = random_trainable model_args={vae_model_args}")
        else:
            vae_params, vae_model_args, vae_backend, _ = load_vae_init(args.vae_ckpt)
    elif args.hand_prior_source == "decoder_only":
        if is_none_vae_ckpt(args.vae_ckpt):
            raise ValueError("decoder_only requires a pretrained --vae_ckpt; VAE_CKPT=none is not supported")
        vae_params, vae_model_args, vae_backend, _ = load_vae_init(args.vae_ckpt)
        if int(vae_model_args["latent_dim"]) != 2:
            raise ValueError(f"decoder_only requires a 2D VAE latent, got {vae_model_args['latent_dim']}")
    elif args.hand_prior_source == "mlp_direct":
        vae_params = None
        vae_model_args = None
        vae_backend = None
    elif args.hand_prior_source == "vq_codebook":
        vae_params = None
        vae_model_args = {"latent_dim": 1}
        vae_backend = "vq_codebook"
    elif args.hand_prior_source == "pca_raw":
        vae_params = None
        vae_model_args = None
        vae_backend = "pca_raw"
    else:
        raise ValueError(f"Unknown hand_prior_source: {args.hand_prior_source}")
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.total_steps, warmup_steps=args.warmup_steps)
    model, state, model_args, has_batch_stats = create_train_state(
        args,
        vae_params,
        vae_model_args,
        lr_schedule,
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
    )
    print(f"Policy parameters: total={count_params(state.params):,}")
    latent_dim = int(model.latent_dim)

    train_step = make_train_step(
        model,
        args.arm_weight,
        args.hand_weight,
        args.hand_loss_space,
        args.reg_drift,
        args.freeze_backbone,
        has_batch_stats,
        args.hand_prior_source,
        args.hand_index_weight,
    )
    eval_step = make_eval_step(model, has_batch_stats, args.hand_prior_source)
    rng = jax.random.PRNGKey(args.seed)
    start_step = 0
    if args.resume_ckpt:
        payload = load_resume_payload(args.resume_ckpt)
        resume_model_args = payload.get("model_args", {})
        resume_image_keys = tuple(
            resume_model_args.get(
                "bc_image_keys",
                resume_model_args.get("image_keys", ("global", "wrist")),
            )
        )
        resume_num_views = int(
            resume_model_args.get("num_image_views", len(resume_image_keys))
        )
        if resume_num_views != int(model_args.get("num_image_views", 2)):
            raise ValueError(
                "Cannot resume BC checkpoint with a different number of image views: "
                f"resume has {resume_num_views} {resume_image_keys}, current run has "
                f"{model_args.get('num_image_views')} {model_args.get('bc_image_keys')}."
            )
        resume_backbone = resume_model_args.get("backbone_type")
        current_backbone = model_args.get("backbone_type")
        if resume_backbone is not None and resume_backbone != current_backbone:
            raise ValueError(
                "Cannot resume BC checkpoint with a different visual backbone: "
                f"resume has {resume_backbone!r}, current run uses {current_backbone!r}. "
                "Pass the matching --backbone_impl for the checkpoint you are resuming."
            )
        resume_hand_prior = resume_model_args.get("hand_prior_source")
        current_hand_prior = model_args.get("hand_prior_source")
        if resume_hand_prior is not None and resume_hand_prior != current_hand_prior:
            raise ValueError(
                "Cannot resume BC checkpoint with a different hand prior source: "
                f"resume has {resume_hand_prior!r}, current run uses {current_hand_prior!r}. "
                "Pass the matching --hand_prior_source for the checkpoint you are resuming."
            )
        resume_step = int(payload.get("step", -1))
        if resume_step < 0:
            raise ValueError(f"Resume checkpoint is missing a valid step: {args.resume_ckpt}")
        if resume_step >= args.total_steps:
            raise ValueError(
                f"Resume checkpoint step ({resume_step}) is already >= total_steps ({args.total_steps})"
            )
        state = state.replace(
            params=payload["params"],
            batch_stats=payload.get("batch_stats", state.batch_stats),
            opt_state=payload.get("opt_state", state.opt_state),
        )
        if "rng" in payload:
            rng = jnp.asarray(payload["rng"])
        start_step = resume_step + 1
        print(f"Resumed from {args.resume_ckpt} at step {resume_step}; continuing from step {start_step}.")
    history = {
        "steps": [],
        "train_total": [],
        "train_arm": [],
        "train_hand": [],
        "train_hand_action": [],
        "train_hand_latent": [],
        "train_weighted_arm": [],
        "train_weighted_hand": [],
        "train_drift": [],
        "train_weighted_drift": [],
        "train_lr": [],
        "eval_steps": [],
        "val_arm": [],
        "val_hand_full": [],
        "val_hand_no_corr": [],
        "val_total": [],
    }

    train_iter = iter(train_loader)
    print(f"Training for steps {start_step}..{args.total_steps - 1}  (train={len(train_ds)}, test={len(test_ds)})\n")
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch_np = torch_batch_to_numpy(batch)
        gt_hand_index_norm = batch_np.get(
            "gt_hand_index_norm",
            np.zeros((batch_np["gt_action"].shape[0], 1), dtype=np.float32),
        )
        gt_delta_z = batch_np.get(
            "gt_delta_z",
            np.zeros((batch_np["gt_action"].shape[0], latent_dim), dtype=np.float32),
        )
        rng, dropout_key = jax.random.split(rng)
        state, metrics = train_step(
            state,
            jnp.asarray(batch_np["img_main"]),
            jnp.asarray(batch_np["img_extra"]),
            extra_images_from_batch(batch_np),
            jnp.asarray(batch_np["state"]),
            jnp.asarray(policy_hand_window_from_batch(batch_np, args.hand_prior_source)),
            jnp.asarray(batch_np["gt_action"]),
            jnp.asarray(gt_delta_z),
            jnp.asarray(gt_hand_index_norm),
            dropout_key,
        )

        base_lr = float(lr_schedule[step])
        history["steps"].append(step)
        history["train_total"].append(float(metrics["total"]))
        history["train_arm"].append(float(metrics["arm"]))
        history["train_hand"].append(float(metrics["decoded_hand"]))
        history["train_hand_action"].append(float(metrics["hand_action"]))
        history["train_hand_latent"].append(float(metrics["hand_latent"]))
        history["train_weighted_arm"].append(float(metrics["weighted_arm"]))
        history["train_weighted_hand"].append(float(metrics["weighted_hand"]))
        history["train_drift"].append(float(metrics["drift"]))
        history["train_weighted_drift"].append(float(metrics["weighted_drift"]))
        history["train_lr"].append(base_lr)

        if step % args.print_freq == 0:
            elapsed = time.time() - t0
            print(
                f"[Step {step:>5d}] total={float(metrics['total']):.6f}  "
                f"arm={float(metrics['arm']):.6f}  hand={float(metrics['decoded_hand']):.6f}  "
                f"hand_loss={float(metrics['hand']):.6f}  "
                f"hand_latent={float(metrics['hand_latent']):.6f}  "
                f"w_arm={float(metrics['weighted_arm']):.6f}  "
                f"w_hand={float(metrics['weighted_hand']):.6f}  "
                f"drift={float(metrics['drift']):.6f}  lr={base_lr:.2e}  ({elapsed:.0f}s)"
            )

        if step > 0 and step % args.eval_freq == 0:
            ev = evaluate_model(state, eval_step, test_loader, args.hand_prior_source)
            history["eval_steps"].append(step)
            history["val_arm"].append(ev["arm_mse"])
            history["val_hand_full"].append(ev["hand_mse_full"])
            history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])
            history["val_total"].append(ev["total_mse"])
            print(
                f"           val arm={ev['arm_mse']:.6f}  "
                f"hand={ev['hand_mse_full']:.6f}  "
                f"no_corr={ev['hand_mse_no_correction']:.6f}  "
                f"vision_gain={ev['hand_mse_no_correction'] - ev['hand_mse_full']:+.6f}"
            )

        if step > 0 and step % args.save_freq == 0:
            saved_path = save_checkpoint(
                state,
                rng,
                step,
                args.output_dir,
                model_args,
                args,
                action_mean,
                action_std,
                vae_backend,
                hand_codebook=hand_codebook,
                hand_pca_mean=hand_pca_mean,
                hand_pca_components=hand_pca_components,
            )
            print(f"           checkpoint saved -> {saved_path}")

    final_ckpt = save_checkpoint(
        state,
        rng,
        args.total_steps,
        args.output_dir,
        model_args,
        args,
        action_mean,
        action_std,
        vae_backend,
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
    )
    ev = evaluate_model(state, eval_step, test_loader, args.hand_prior_source)
    history["eval_steps"].append(args.total_steps)
    history["val_arm"].append(ev["arm_mse"])
    history["val_hand_full"].append(ev["hand_mse_full"])
    history["val_hand_no_corr"].append(ev["hand_mse_no_correction"])
    history["val_total"].append(ev["total_mse"])

    final_metrics = {
        "arm_mse": ev["arm_mse"],
        "hand_mse_full": ev["hand_mse_full"],
        "hand_mse_no_correction": ev["hand_mse_no_correction"],
        "hand_index_mse": ev["hand_index_mse"],
        "hand_index_acc": ev["hand_index_acc"],
        "total_mse": ev["total_mse"],
        "vision_gain": ev["hand_mse_no_correction"] - ev["hand_mse_full"],
    }
    if args.hand_prior_source == "vq_codebook":
        print(
            f"\nFinal val: arm={ev['arm_mse']:.6f}  decoded_hand={ev['hand_mse_full']:.6f}  "
            f"index_mse={ev['hand_index_mse']:.6f}  index_acc={ev['hand_index_acc']:.3f}"
        )
    else:
        print(
            f"\nFinal val: arm={ev['arm_mse']:.6f}  hand={ev['hand_mse_full']:.6f}  "
            f"no_corr={ev['hand_mse_no_correction']:.6f}  vision_gain={final_metrics['vision_gain']:+.6f}"
        )

    save_training_curves(history, args.output_dir, args)
    save_summary(history, args.output_dir, args, model_args, final_metrics, action_mean, action_std, vae_backend)
    print(f"\nDone! JAX checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()

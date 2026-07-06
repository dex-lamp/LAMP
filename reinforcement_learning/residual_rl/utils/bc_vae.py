"""Utilities for BCVAE checkpoint loading, preprocessing, and action adaptation.

The BCVAE model itself lives in `reinforcement_learning.residual_rl.networks.bc_vae`; this file owns the
runtime glue around that model:

* locate and read JAX checkpoint payloads,
* validate the metadata expected by residual SAC and TD3 head finetuning,
* construct `BCVAEPolicy` instances from saved `model_args`,
* convert SERL observations into BCVAE tensor inputs, and
* convert BCVAE semantic outputs into the 12D environment action format.

Keeping this logic here prevents training scripts and RL agents from growing
their own incompatible copies of image/state normalization or checkpoint
schema handling.
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np

from reinforcement_learning.residual_rl.networks.bc_vae import BCVAEPolicy


@dataclass(frozen=True)
class BCVAECheckpoint:
    """Loaded BCVAE checkpoint plus derived metadata.

    Fields are intentionally explicit because different callers need different
    views of the same checkpoint: pure inference uses `variables`, TD3 head
    finetuning needs raw `params`, and all paths need `action_mean/std` to
    reproduce BC training-time state normalization.
    """

    model: BCVAEPolicy
    variables: flax.core.FrozenDict
    params: flax.core.FrozenDict
    batch_stats: flax.core.FrozenDict
    payload: dict[str, Any]
    checkpoint_dir: Path
    action_mean: np.ndarray
    action_std: np.ndarray
    hand_history_std_floor: float
    window_size: int
    bc_image_keys: tuple[str, ...]
    num_image_views: int
    hand_prior_source: str
    latent_dim: int
    hand_core_dim: int
    core_action_dim: int


def resolve_path(path: str | os.PathLike[str], base_dir: Optional[Path] = None) -> Path:
    """Resolve a user/checkpoint path relative to `base_dir` or the CWD."""
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    return (base_dir or Path.cwd()) / out


def latest_jax_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> Optional[str]:
    """Return the latest `checkpoint*.pkl` file in a BCVAE checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    latest_path = checkpoint_path / "checkpoint.pkl"
    if latest_path.exists():
        return str(latest_path)
    candidates = glob.glob(os.path.join(str(checkpoint_path), "checkpoint_*.pkl"))
    candidates.sort(key=lambda path: int(Path(path).stem.split("_")[-1]))
    return candidates[-1] if candidates else None


def save_jax_checkpoint(
    checkpoint_dir: str | os.PathLike[str], payload: dict[str, Any], step: int
) -> str:
    """Save a BCVAE-compatible JAX checkpoint payload.

    The function writes both a step-specific file and `checkpoint.pkl`, matching
    the layout used by the external BC/VAE training code.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["step"] = int(step)
    step_path = checkpoint_path / f"checkpoint_{step}.pkl"
    latest_path = checkpoint_path / "checkpoint.pkl"
    with step_path.open("wb") as f:
        pickle.dump(payload, f)
    with latest_path.open("wb") as f:
        pickle.dump(payload, f)
    return str(step_path)


def restore_jax_checkpoint(
    checkpoint_dir: str | os.PathLike[str], step: Optional[int] = None
) -> dict[str, Any]:
    """Load a BCVAE-compatible JAX checkpoint payload from disk."""
    checkpoint_path = Path(checkpoint_dir)
    if step is None:
        latest = latest_jax_checkpoint(checkpoint_path)
        if latest is None:
            raise FileNotFoundError(f"No JAX checkpoint found in {checkpoint_dir}")
        ckpt_file = Path(latest)
    else:
        ckpt_file = checkpoint_path / f"checkpoint_{step}.pkl"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"JAX checkpoint not found: {ckpt_file}")
    with ckpt_file.open("rb") as f:
        return pickle.load(f)


def resolve_jax_checkpoint_dir(root: str | os.PathLike[str]) -> Path:
    """Resolve a checkpoint root or its nested `jax_checkpoints` directory."""
    root_path = Path(root).expanduser()
    if (root_path / "checkpoint.pkl").exists():
        return root_path
    nested = root_path / "jax_checkpoints"
    if (nested / "checkpoint.pkl").exists():
        return nested
    raise FileNotFoundError(f"Could not resolve JAX checkpoint dir from: {root}")


def _load_hand_codebook(model_args: dict[str, Any], payload: dict[str, Any]) -> np.ndarray | None:
    """Load optional VQ hand codebook from payload or its saved path."""
    if model_args.get("hand_prior_source") != "vq_codebook":
        return None
    if "hand_codebook" in payload:
        return np.asarray(payload["hand_codebook"], dtype=np.float32)
    hand_codebook_path = model_args.get("hand_codebook_path")
    if not hand_codebook_path:
        raise ValueError(
            "VQ BCVAE checkpoint is missing both payload['hand_codebook'] "
            "and model_args['hand_codebook_path']."
        )
    return np.load(resolve_path(hand_codebook_path)).astype(np.float32)


def _load_hand_pca_params(model_args: dict[str, Any], payload: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load PCA hand decoder parameters embedded in the checkpoint payload."""
    if model_args.get("hand_prior_source") != "pca_raw":
        return None, None
    if "hand_pca_mean" not in payload or "hand_pca_components" not in payload:
        raise ValueError(
            "pca_raw BCVAE checkpoint must embed payload['hand_pca_mean'] "
            "and payload['hand_pca_components']."
        )
    mean = np.asarray(payload["hand_pca_mean"], dtype=np.float32)
    components = np.asarray(payload["hand_pca_components"], dtype=np.float32)
    if mean.shape != (6,) or components.ndim != 2 or components.shape[0] < 2 or components.shape[1] != 6:
        raise ValueError(
            f"Invalid pca_raw PCA params: mean={mean.shape}, components={components.shape}; "
            "expected mean (6,) and components at least (2, 6)."
        )
    return mean, components


def _validate_direct_head_schema(model_args: dict[str, Any]) -> None:
    """Validate direct-head checkpoints before robot-side JIT/apply."""
    hand_prior_source = model_args.get("hand_prior_source")
    if hand_prior_source == "mlp_direct":
        expected = "core_action_encoded_normalized_hand_history_direct_12d"
        expected_description = "`visual+hand_encoder(flatten(normalized_hand_history))+arm_state_encoder(arm_state_6d) -> action_12d`"
        expected_latent_dim = 6
    elif hand_prior_source == "decoder_only":
        expected = "core_action_encoded_normalized_hand_history_decoder_only_8d"
        expected_description = "`visual+hand_encoder(flatten(normalized_hand_history))+arm_state_encoder(arm_state_6d) -> [arm_6d,z_2d]`"
        expected_latent_dim = 2
    elif hand_prior_source == "vq_codebook":
        expected = "core_action_encoded_normalized_hand_history_vq_codebook_7d"
        expected_description = "`visual+hand_encoder(flatten(normalized_hand_history))+arm_state_encoder(arm_state_6d) -> [arm_6d,index_1d]`"
        expected_latent_dim = 1
    elif hand_prior_source == "pca_raw":
        expected = "core_action_encoded_raw_pca_history_pca_raw_8d"
        expected_description = "`visual+hand_encoder(flatten(PCA2(raw_hand_history)))+arm_state_encoder(arm_state_6d) -> [arm_6d,pca_z_2d]`"
        expected_latent_dim = 2
    else:
        return
    actual = model_args.get("direct_head_type")
    latent_dim = int(model_args.get("vae", {}).get("latent_dim", 6))
    if hand_prior_source == "pca_raw":
        latent_dim = int(model_args.get("hand_pca_dim", 0))
    if (
        actual != expected
        or int(model_args.get("hand_state_dim", 0)) != 6
        or (
            hand_prior_source in {"decoder_only", "vq_codebook", "pca_raw"}
            and latent_dim != expected_latent_dim
        )
    ):
        raise ValueError(
            f"Unsupported {hand_prior_source} BCVAE checkpoint schema. Online "
            "inference expects a single direct core action head "
            f"{expected_description}, with "
            f"model_args['direct_head_type']={expected!r} and "
            "model_args['hand_state_dim']=6. "
            f"Got direct_head_type={actual!r}, "
            f"hand_state_dim={model_args.get('hand_state_dim')!r}, "
            f"latent_dim={latent_dim!r}. "
            "Retrain/export the BC checkpoint with the updated behavior_clone "
            f"{hand_prior_source} architecture."
        )


def build_bc_vae_policy_from_payload(payload: dict[str, Any]) -> BCVAEPolicy:
    """Construct a `BCVAEPolicy` using the checkpoint's saved model metadata."""
    model_args = payload["model_args"]
    _validate_direct_head_schema(model_args)
    hand_codebook = _load_hand_codebook(model_args, payload)
    hand_pca_mean, hand_pca_components = _load_hand_pca_params(model_args, payload)
    return BCVAEPolicy(
        vae_model_args=model_args.get("vae"),
        hand_prior_source=model_args.get("hand_prior_source", "vae"),
        hand_codebook=hand_codebook,
        hand_pca_mean=hand_pca_mean,
        hand_pca_components=hand_pca_components,
        hand_pca_dim=int(model_args.get("hand_pca_dim", 2)),
        window_size=int(model_args.get("window_size", 8)),
        num_image_views=int(
            model_args.get(
                "num_image_views",
                len(
                    model_args.get(
                        "bc_image_keys",
                        model_args.get("image_keys", ("global", "wrist")),
                    )
                ),
            )
        ),
        backbone_type=model_args.get("backbone_type", "hf_resnet18"),
        backbone_config=model_args.get("backbone_config"),
        state_encoder_type=model_args["state_encoder"],
        arm_state_dim=model_args.get("arm_state_dim", 6),
        hand_state_dim=model_args.get("hand_state_dim", 6),
        dropout=model_args["dropout"],
        hil_serl_root=model_args.get("hil_serl_root") or os.environ.get("HIL_SERL_ROOT", ""),
        hil_serl_pooling_method=model_args.get(
            "hil_serl_pooling_method", "spatial_learned_embeddings"
        ),
        hil_serl_num_spatial_blocks=model_args.get("hil_serl_num_spatial_blocks", 8),
        hil_serl_bottleneck_dim=model_args.get("hil_serl_bottleneck_dim", 256),
    )


def _validate_action_stats(payload: dict[str, Any], action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract and validate per-dimension BC state normalization statistics."""
    action_mean = np.asarray(payload["action_mean"], dtype=np.float32)
    action_std = np.asarray(payload["action_std"], dtype=np.float32)
    if action_mean.shape != (action_dim,) or action_std.shape != (action_dim,):
        raise ValueError(
            f"Expected BCVAE action_mean/action_std shape ({action_dim},), got "
            f"{action_mean.shape}/{action_std.shape}"
        )
    return action_mean, action_std


def _resolve_hand_history_std_floor(payload: dict[str, Any]) -> float:
    """Return the std floor used for normalized direct hand-history inputs."""
    model_args = payload.get("model_args", {})
    train_args = payload.get("train_args", {})
    floor = float(model_args.get("hand_history_std_floor", train_args.get("hand_history_std_floor", 1e-6)))
    if floor < 0:
        raise ValueError(f"hand_history_std_floor must be >= 0, got {floor}")
    return floor


def load_bc_vae_checkpoint(
    ckpt_root: str | os.PathLike[str],
    *,
    require_hand_prior_source: str | None = None,
    require_latent_dim: int | None = None,
    action_dim: int = 12,
) -> BCVAECheckpoint:
    """Load and validate a BCVAE checkpoint.

    Args:
        ckpt_root: Checkpoint dir, run dir containing `jax_checkpoints`, or path
            relative to the caller's current working directory.
        require_hand_prior_source: Optional schema guard for callers that only
            support a single BC hand-prior family.
        require_latent_dim: Optional latent dimension guard, e.g. `2`.
        action_dim: Expected dimension of action statistics and env actions.

    Returns:
        `BCVAECheckpoint` with a constructed model, frozen variables, raw
        params/batch_stats, action statistics, and model metadata.
    """
    ckpt_dir = resolve_jax_checkpoint_dir(resolve_path(ckpt_root))
    payload = restore_jax_checkpoint(ckpt_dir)
    model_args = payload["model_args"]
    bc_image_keys = tuple(
        model_args.get("bc_image_keys", model_args.get("image_keys", ("global", "wrist")))
    )
    num_image_views = int(model_args.get("num_image_views", len(bc_image_keys)))
    if num_image_views < 2:
        raise ValueError(f"BCVAE requires at least two image views, got {num_image_views}")
    if len(bc_image_keys) != num_image_views:
        bc_image_keys = tuple(f"image_{idx}" for idx in range(num_image_views))
    hand_prior_source = model_args.get("hand_prior_source", "vae")
    if hand_prior_source == "vae":
        latent_dim = int(model_args["vae"]["latent_dim"])
    elif hand_prior_source == "vq_codebook":
        latent_dim = 1
    elif hand_prior_source == "mlp_direct":
        latent_dim = 6
    elif hand_prior_source == "decoder_only":
        latent_dim = int(model_args["vae"]["latent_dim"])
        if latent_dim != 2:
            raise ValueError(f"decoder_only requires latent_dim=2, got {latent_dim}")
    elif hand_prior_source == "pca_raw":
        latent_dim = int(model_args.get("hand_pca_dim", 2))
        if latent_dim != 2:
            raise ValueError(f"pca_raw requires hand_pca_dim=2, got {latent_dim}")
    else:
        raise ValueError(f"Unknown hand_prior_source={hand_prior_source!r}")

    if require_hand_prior_source and hand_prior_source != require_hand_prior_source:
        raise ValueError(
            f"Expected hand_prior_source={require_hand_prior_source!r}, "
            f"got {hand_prior_source!r}"
        )
    if require_latent_dim is not None and latent_dim != int(require_latent_dim):
        raise ValueError(
            f"Expected BCVAE latent_dim={require_latent_dim}, got {latent_dim}."
        )

    hand_core_dim = 6 if hand_prior_source == "mlp_direct" else latent_dim
    core_action_dim = 6 + hand_core_dim

    model = build_bc_vae_policy_from_payload(payload)
    params = flax.core.freeze(payload["params"])
    batch_stats = flax.core.freeze(payload.get("batch_stats", {}))
    variables = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats
    action_mean, action_std = _validate_action_stats(payload, action_dim)
    hand_history_std_floor = _resolve_hand_history_std_floor(payload)

    return BCVAECheckpoint(
        model=model,
        variables=flax.core.freeze(variables),
        params=params,
        batch_stats=batch_stats,
        payload=payload,
        checkpoint_dir=ckpt_dir,
        action_mean=action_mean,
        action_std=action_std,
        hand_history_std_floor=hand_history_std_floor,
        window_size=int(model_args.get("window_size", 8)),
        bc_image_keys=bc_image_keys,
        num_image_views=num_image_views,
        hand_prior_source=hand_prior_source,
        latent_dim=latent_dim,
        hand_core_dim=hand_core_dim,
        core_action_dim=core_action_dim,
    )


def make_variables(params, batch_stats=None) -> flax.core.FrozenDict:
    """Build a Flax variables dict from params and optional batch statistics."""
    variables = {"params": params}
    if batch_stats:
        variables["batch_stats"] = batch_stats
    return flax.core.freeze(variables)


def _ensure_batched(x: jax.Array, batched: bool) -> jax.Array:
    """Add a batch dimension for online single-observation inference."""
    x = jnp.asarray(x)
    return x if batched else x[None, ...]


def prepare_bc_vae_image(image: jax.Array, batched: bool) -> jax.Array:
    """Convert SERL HWC image observations to BCVAE NCHW float images.

    Replay batches are usually `(B, T, H, W, C)` because `ChunkingWrapper`
    stacks observations.  Online observations are usually `(T, H, W, C)`.  The
    BC policy consumes the latest frame only, in NCHW layout and `[0, 1]`.
    """
    image = jnp.asarray(image)
    if batched:
        if image.ndim == 5:
            image = image[:, -1, ...]
        elif image.ndim != 4:
            raise ValueError(f"Expected batched image with 4 or 5 dims, got {image.shape}")
    else:
        if image.ndim == 4:
            image = image[-1, ...][None, ...]
        elif image.ndim == 3:
            image = image[None, ...]
        else:
            raise ValueError(f"Expected unbatched image with 3 or 4 dims, got {image.shape}")

    image = image.astype(jnp.float32)
    image = image / jnp.where(jnp.max(image) > 1.5, 255.0, 1.0)
    return jnp.transpose(image, (0, 3, 1, 2))


def _edge_random_crop_nchw(image: jax.Array, crop_from: jax.Array, padding: int) -> jax.Array:
    """Edge-padded random-shift crop for one NCHW image, output size unchanged."""
    if padding <= 0:
        return image
    _, height, width = image.shape
    padded = jnp.pad(
        image,
        (
            (0, 0),
            (padding, padding),
            (padding, padding),
        ),
        mode="edge",
    )
    start = (0, crop_from[0], crop_from[1])
    return jax.lax.dynamic_slice(padded, start, image.shape)


def paired_edge_random_crop_nchw(
    img_main: jax.Array,
    img_extra: jax.Array,
    rng: jax.Array,
    padding: int,
) -> tuple[jax.Array, jax.Array]:
    """Apply the same edge-padded random crop offset to both BCVAE image views."""
    if padding <= 0:
        return img_main, img_extra
    crop_from = jax.random.randint(
        rng,
        (img_main.shape[0], 2),
        0,
        2 * padding + 1,
        dtype=jnp.int32,
    )
    crop_one = lambda image, offset: _edge_random_crop_nchw(image, offset, padding)
    return (
        jax.vmap(crop_one, in_axes=(0, 0), out_axes=0)(img_main, crop_from),
        jax.vmap(crop_one, in_axes=(0, 0), out_axes=0)(img_extra, crop_from),
    )


def multi_edge_random_crop_nchw(
    images: tuple[jax.Array, ...],
    rng: jax.Array,
    padding: int,
) -> tuple[jax.Array, ...]:
    """Apply the same edge-padded random-shift crop offset to N NCHW image views."""
    if padding <= 0 or not images:
        return images
    crop_from = jax.random.randint(
        rng,
        (images[0].shape[0], 2),
        0,
        2 * padding + 1,
        dtype=jnp.int32,
    )
    crop_one = lambda image, offset: _edge_random_crop_nchw(image, offset, padding)
    return tuple(
        jax.vmap(crop_one, in_axes=(0, 0), out_axes=0)(image, crop_from)
        for image in images
    )


def is_batched_bc_vae_observation(observations: dict[str, Any]) -> bool:
    """Return whether an observation tree has a leading batch dimension."""
    return jnp.asarray(observations["bc_policy_state"]).ndim >= 2


def prepare_bc_vae_policy_inputs(
    observations: dict[str, Any],
    *,
    action_mean: jax.Array,
    action_std: jax.Array,
    image_keys: Iterable[str] = ("global", "wrist"),
    require_hand_history: bool = True,
    normalize_hand_history: bool = False,
    hand_history_std_floor: float = 1e-6,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 4,
    crop_rng: Optional[jax.Array] = None,
):
    """Convert an augmented SERL observation into BCVAE policy inputs.

    Returns:
        `(img_main, img_extra, extra_images, state, past_hand_win,
        current_hand_abs, batched)`.
        The first five values are direct inputs to `BCVAEPolicy.__call__`;
        `current_hand_abs` is used by action adapters to convert absolute hand
        predictions into environment hand deltas.
    """
    batched = is_batched_bc_vae_observation(observations)
    image_keys = tuple(image_keys)
    if len(image_keys) < 2:
        raise ValueError(f"BCVAE requires at least two image keys, got {image_keys}")

    # Image conversion is the bridge between SERL env/replay format and the BC
    # training format.  Keeping it centralized avoids mismatched channels/layout.
    missing_image_keys = [key for key in image_keys if key not in observations]
    if missing_image_keys:
        raise KeyError(
            f"BCVAE observation is missing image key(s) {missing_image_keys}; "
            f"requested image_keys={image_keys}."
        )
    images = tuple(
        prepare_bc_vae_image(observations[image_key], batched)
        for image_key in image_keys
    )
    if random_crop_padding < 0:
        raise ValueError(f"random_crop_padding must be >= 0, got {random_crop_padding}")
    if enable_image_random_crop:
        if crop_rng is None:
            crop_rng = jax.random.PRNGKey(0)
        images = multi_edge_random_crop_nchw(
            images,
            crop_rng,
            int(random_crop_padding),
        )
    img_main, img_extra = images[:2]
    extra_images = tuple(images[2:])

    # The BC policy was trained with z-scored 12D policy_state values.  These
    # statistics come from the BC training dataset and are saved in the payload.
    policy_state = _ensure_batched(observations["bc_policy_state"], batched)
    safe_std = jnp.maximum(jnp.asarray(action_std), 1e-6)
    state = ((policy_state - jnp.asarray(action_mean)) / safe_std).astype(jnp.float32)

    if require_hand_history:
        past_hand_win = _ensure_batched(observations["past_hand_win"], batched)
        if normalize_hand_history:
            hand_mean = jnp.asarray(action_mean)[6:12]
            floor = jnp.maximum(jnp.asarray(hand_history_std_floor, dtype=jnp.float32), 1e-6)
            hand_std = jnp.maximum(jnp.asarray(action_std)[6:12], floor)
            past_hand_win = ((past_hand_win - hand_mean) / hand_std).astype(jnp.float32)
    else:
        past_hand_win = None
    current_hand_abs = _ensure_batched(observations["current_hand_abs"], batched)
    return (
        img_main,
        img_extra,
        extra_images,
        state,
        past_hand_win,
        current_hand_abs,
        batched,
    )


def prepare_bc_vae_inputs(
    observations: dict[str, Any],
    *,
    action_mean: jax.Array,
    action_std: jax.Array,
    image_keys: Iterable[str] = ("global", "wrist"),
    require_hand_history: bool = True,
    normalize_hand_history: bool = False,
    hand_history_std_floor: float = 1e-6,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 4,
    crop_rng: Optional[jax.Array] = None,
):
    """Convert observations into the legacy two-image BCVAE input tuple.

    Existing algorithm paths call this helper and expect exactly `(img_main,
    img_extra, state, past_hand_win, current_hand_abs, batched)`. Keep that
    contract stable while
    `prepare_bc_vae_policy_inputs` handles configurable multi-view BC policy
    evaluation.
    """
    image_keys = tuple(image_keys)
    if len(image_keys) < 2:
        raise ValueError(f"BCVAE requires at least two image keys, got {image_keys}")
    (
        img_main,
        img_extra,
        _extra_images,
        state,
        past_hand_win,
        current_hand_abs,
        batched,
    ) = prepare_bc_vae_policy_inputs(
        observations,
        action_mean=action_mean,
        action_std=action_std,
        image_keys=image_keys[:2],
        require_hand_history=require_hand_history,
        normalize_hand_history=normalize_hand_history,
        hand_history_std_floor=hand_history_std_floor,
        enable_image_random_crop=enable_image_random_crop,
        random_crop_padding=random_crop_padding,
        crop_rng=crop_rng,
    )
    return img_main, img_extra, state, past_hand_win, current_hand_abs, batched


def bc_vae_outputs_to_env_actions(
    outputs: dict[str, jax.Array],
    current_hand_abs: jax.Array,
    *,
    action_low: jax.Array,
    action_high: jax.Array,
    clip: bool,
    batched: bool,
) -> jax.Array:
    """Convert BCVAE semantic outputs into 12D environment actions.

    BCVAE predicts an absolute hand pose.  The real environment expects a hand
    delta, so this adapter computes `hand_action_abs - current_hand_abs` and
    concatenates it with the 6D arm command.
    """
    arm_action = outputs["arm_action"]
    hand_delta = outputs["hand_action"] - current_hand_abs
    env_actions = jnp.concatenate([arm_action, hand_delta], axis=-1)
    if clip:
        env_actions = jnp.clip(env_actions, action_low, action_high)
    return env_actions if batched else env_actions[0]


def export_bc_vae_checkpoint_from_td3_agent(agent, payload: dict[str, Any], checkpoint_path: str, step: int) -> str:
    """Export a TD3-head agent back into a directly evaluable BCVAE checkpoint.

    `BCHeadTD3Agent` stores trainable action heads in its actor params and keeps
    the rest of the BCVAE checkpoint frozen.  This helper merges those heads
    into the original BCVAE payload so `eval_bc_vae.py` can run the finetuned
    policy without TD3-specific code.
    """
    export_dir = Path(checkpoint_path) / "exported_bc_vae"
    base_params = flax.core.unfreeze(jax.device_get(agent.bc_params))
    actor_heads = flax.core.unfreeze(jax.device_get(agent.state.params["actor"]))
    base_params["arm_head"] = actor_heads["arm_head"]
    base_params["hand_delta_z_head"] = actor_heads["hand_delta_z_head"]

    export_payload = dict(payload)
    export_payload["params"] = flax.core.freeze(base_params)
    if agent.config["has_batch_stats"]:
        export_payload["batch_stats"] = jax.device_get(agent.batch_stats)
    elif "batch_stats" in export_payload:
        export_payload.pop("batch_stats")

    return save_jax_checkpoint(str(export_dir), export_payload, step=step)

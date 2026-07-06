"""Factory functions for public residual SAC/RLPD agents.

The private robot launcher also owns actor/learner networking, environment
construction, logging, and checkpoint scheduling.  This module deliberately
keeps only the algorithm/model construction entry point.
"""

from __future__ import annotations

from typing import Callable, Optional
import os

import jax
from jax import nn

from reinforcement_learning.residual_rl.agents.continuous.residual_sac import (
    ResidualSACAgent,
)
from reinforcement_learning.residual_rl.common.typing import Batch, PRNGKey
from reinforcement_learning.residual_rl.utils.bc_vae import BCVAECheckpoint
from reinforcement_learning.residual_rl.vision.data_augmentations import (
    batched_random_crop,
)


def make_batch_augmentation_func(image_keys, *, padding: int = 4) -> Callable:
    """Create edge-padded random crop augmentation for replay batches."""

    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key],
                        rng,
                        padding=padding,
                        num_batch_dims=2,
                    )
                }
            )
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        _, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        return batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

    return augment_batch


def maybe_make_batch_augmentation_func(
    image_keys,
    *,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 4,
) -> Optional[Callable]:
    """Return image augmentation or ``None`` when disabled."""

    if random_crop_padding < 0:
        raise ValueError(f"random_crop_padding must be >= 0, got {random_crop_padding}")
    if not enable_image_random_crop:
        return None
    return make_batch_augmentation_func(image_keys, padding=random_crop_padding)


def make_residual_sac_pixel_agent(
    rng,
    sample_obs,
    sample_action,
    *,
    action_low,
    action_high,
    bc_payload: BCVAECheckpoint,
    image_keys=("image",),
    bc_image_keys=("global", "wrist"),
    residual_extra_image_keys=(),
    residual_extra_encoder_type="bc-aligned",
    freeze_residual_extra_encoders=True,
    encoder_type="resnet-pretrained",
    discount=0.97,
    residual_scale_arm=0.05,
    residual_scale_core=None,
    residual_scale_z=None,
    target_entropy=None,
    zero_delta=False,
    enable_image_random_crop: bool = False,
    random_crop_padding: int = 4,
):
    """Create a residual SAC agent over a frozen BC policy.

    The residual actor distribution lives in the BC semantic core-action space.
    The critic and replay actions remain in the environment action space
    ``[arm_action_6, hand_delta_6]``.
    """

    if residual_scale_core is None:
        residual_scale_core = 0.05 if residual_scale_z is None else residual_scale_z
    bc_model_args = bc_payload.payload.get("model_args", {})
    bc_hil_serl_root = bc_model_args.get("hil_serl_root") or os.environ.get(
        "HIL_SERL_ROOT",
        "",
    )

    return ResidualSACAgent.create_pixels(
        rng,
        sample_obs,
        sample_action,
        bc_model=bc_payload.model,
        bc_variables=bc_payload.variables,
        action_mean=bc_payload.action_mean,
        action_std=bc_payload.action_std,
        action_low=action_low,
        action_high=action_high,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        bc_image_keys=bc_image_keys,
        residual_extra_image_keys=residual_extra_image_keys,
        residual_extra_encoder_type=residual_extra_encoder_type,
        freeze_residual_extra_encoders=freeze_residual_extra_encoders,
        bc_backbone_type=bc_model_args.get("backbone_type", "hf_resnet18"),
        bc_backbone_config=bc_model_args.get("backbone_config"),
        bc_backbone_params=bc_payload.params.get("backbone"),
        bc_backbone_batch_stats=bc_payload.batch_stats.get("backbone", {}),
        bc_hil_serl_root=bc_hil_serl_root,
        bc_hil_serl_pooling_method=bc_model_args.get(
            "hil_serl_pooling_method",
            "spatial_learned_embeddings",
        ),
        bc_hil_serl_num_spatial_blocks=bc_model_args.get(
            "hil_serl_num_spatial_blocks",
            8,
        ),
        bc_hil_serl_bottleneck_dim=bc_model_args.get("hil_serl_bottleneck_dim", 256),
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        target_entropy=(
            0.5 * float(bc_payload.core_action_dim)
            if target_entropy is None
            else float(target_entropy)
        ),
        augmentation_function=maybe_make_batch_augmentation_func(
            image_keys,
            enable_image_random_crop=enable_image_random_crop,
            random_crop_padding=random_crop_padding,
        ),
        residual_action_dim=bc_payload.core_action_dim,
        hand_prior_source=bc_payload.hand_prior_source,
        hand_core_dim=bc_payload.hand_core_dim,
        core_action_dim=bc_payload.core_action_dim,
        residual_scale_arm=residual_scale_arm,
        residual_scale_core=residual_scale_core,
        zero_delta=zero_delta,
        normalize_hand_history=(
            bc_payload.hand_prior_source in {"mlp_direct", "decoder_only", "vq_codebook"}
        ),
        hand_history_std_floor=bc_payload.hand_history_std_floor,
    )

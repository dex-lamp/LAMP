"""BC policy that combines visual features with a VAE hand-action prior.

This public supplement copy contains the pure Flax model layer used by
residual RL:

* Inputs are already preprocessed tensors: NCHW image batches in `[0, 1]`,
  a normalized 12D policy state, and an optional hand-history window.
* Outputs are semantic policy components: 6D arm action, absolute 6D hand
  action, latent hand control values, and debug features.
* No checkpoints, replay buffers, robot actions, or actor/learner loops live
  here. Those responsibilities are handled by the surrounding residual RL
  utilities and by user-provided environments.
"""

import importlib
import os
import sys
from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

try:
    from transformers import ResNetConfig
    from transformers.models.resnet.modeling_flax_resnet import FlaxResNetModule
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on minimal envs.
    ResNetConfig = None
    FlaxResNetModule = None
    _TRANSFORMERS_IMPORT_ERROR = exc

from reinforcement_learning.residual_rl.networks.vae import HandActionVAE


Array = jax.Array

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_HIL_SERL_ROOT = os.environ.get("HIL_SERL_ROOT", "")


def _load_hil_serl_resnet_v1(hil_serl_root: str):
    """Load the SERL ResNet implementation used by pretrained visual checkpoints.

    The supplement package vendors a compatible
    `reinforcement_learning.residual_rl.vision.resnet_v1`. We prefer that local
    implementation so BCVAE does not depend on a private external checkout. The
    `hil_serl_root` argument is kept in the model signature for checkpoint
    compatibility; if the local import is unavailable, we fall back to the
    historical external path and raise a clear error if neither exists.
    """
    try:
        from reinforcement_learning.residual_rl.vision import resnet_v1

        return resnet_v1.PreTrainedResNetEncoder, resnet_v1.resnetv1_configs
    except Exception:
        launcher_root = os.path.join(hil_serl_root, "serl_launcher")
        if not os.path.isdir(launcher_root):
            raise FileNotFoundError(
                f"HiL-SERL launcher package not found under: {launcher_root}. "
                "The local reinforcement_learning.residual_rl.vision.resnet_v1 import also failed. "
                "Set HIL_SERL_ROOT if you want to use an external HiL-SERL checkout."
            )
        if launcher_root not in sys.path:
            sys.path.insert(0, launcher_root)
        resnet_v1 = importlib.import_module("serl_launcher.vision.resnet_v1")
        return resnet_v1.PreTrainedResNetEncoder, resnet_v1.resnetv1_configs


def _torch_uniform(key, shape, dtype, fan_in):
    """Return PyTorch-like uniform initialization for checkpoint compatibility."""
    bound = 1.0 / jnp.sqrt(float(fan_in))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


class TorchDense(nn.Module):
    """Dense layer with PyTorch-like default initialization and stable param names."""

    features: int
    zero_init: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        in_features = x.shape[-1]
        if self.zero_init:
            kernel_init = lambda key, shape: jnp.zeros(shape, dtype=x.dtype)
            bias_init = lambda key, shape: jnp.zeros(shape, dtype=x.dtype)
        else:
            kernel_init = lambda key, shape: _torch_uniform(
                key, shape, x.dtype, in_features
            )
            bias_init = lambda key, shape: _torch_uniform(
                key, shape, x.dtype, in_features
            )

        kernel = self.param("kernel", kernel_init, (in_features, self.features))
        bias = self.param("bias", bias_init, (self.features,))
        return jnp.matmul(x, kernel) + bias


class StateEncoderMLP(nn.Module):
    """Encode the normalized policy state with a two-layer MLP."""

    hidden_dim: int = 128

    def setup(self) -> None:
        self.layer_0 = TorchDense(self.hidden_dim, name="layer_0")
        self.layer_1 = TorchDense(self.hidden_dim, name="layer_1")

    def __call__(self, x: Array) -> Array:
        h = nn.relu(self.layer_0(x))
        h = nn.relu(self.layer_1(h))
        return h


class StateEncoderLinear64(nn.Module):
    """Small state encoder kept for compatibility with older BC checkpoints."""

    def setup(self) -> None:
        self.layer_0 = TorchDense(64, name="layer_0")

    def __call__(self, x: Array) -> Array:
        return nn.relu(self.layer_0(x))


class StateEncoderRaw(nn.Module):
    """Pass normalized state through unchanged."""

    def __call__(self, x: Array) -> Array:
        return x


def build_state_encoder_module(encoder_type: str):
    """Build the state encoder requested by `model_args["state_encoder"]`."""
    if encoder_type == "mlp":
        return StateEncoderMLP(), 128
    if encoder_type == "linear64":
        return StateEncoderLinear64(), 64
    if encoder_type == "raw":
        return StateEncoderRaw(), None
    raise ValueError(f"Unknown state_encoder: {encoder_type}")


class CoreActionHead(nn.Module):
    """Predict the semantic core action from visual, hand, and arm features."""

    in_dim: int
    out_dim: int
    dropout: float = 0.0

    def setup(self) -> None:
        self.fc1 = TorchDense(512, name="fc1")
        self.fc_extra = TorchDense(512, name="fc_extra")
        self.fc2 = TorchDense(256, name="fc2")
        self.fc_out = TorchDense(self.out_dim, name="fc_out")

    def __call__(self, x: Array, deterministic: bool) -> Array:
        h = nn.relu(self.fc1(x))
        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.relu(self.fc_extra(h))
        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.relu(self.fc2(h))
        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        return self.fc_out(h)


class ResNet18Backbone(nn.Module):
    """HuggingFace Flax ResNet-18 backbone with Torch-style preprocessing."""

    backbone_config: Dict[str, Any]

    def setup(self) -> None:
        if ResNetConfig is None or FlaxResNetModule is None:
            raise ImportError(
                "HF ResNet BCVAE checkpoints require the transformers package."
            ) from _TRANSFORMERS_IMPORT_ERROR
        if self.backbone_config is None:
            raise ValueError("backbone_config is required for hf_resnet18 BCVAE checkpoints.")
        config_dict = dict(self.backbone_config)
        config_dict.pop("id2label", None)
        config_dict.pop("label2id", None)
        config_dict.pop("_name_or_path", None)
        self.resnet = FlaxResNetModule(
            config=ResNetConfig.from_dict(config_dict),
            dtype=jnp.float32,
            name="resnet",
        )

    def __call__(self, x: Array, train_backbone: bool = False) -> Array:
        # HF ResNet expects NHWC images normalized with ImageNet statistics.
        mean = jnp.asarray(IMAGENET_MEAN, dtype=x.dtype).reshape((1, 1, 1, 3))
        std = jnp.asarray(IMAGENET_STD, dtype=x.dtype).reshape((1, 1, 1, 3))
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = (x - mean) / std
        outputs = self.resnet(
            x,
            deterministic=not train_backbone,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.pooler_output.reshape((x.shape[0], -1))


class HiLSerlResNet10Backbone(nn.Module):
    """Multi-camera SERL ResNet-10 visual stack.

    Each camera has its own `PreTrainedResNetEncoder`.  The nested encoder
    structure and names match the training checkpoint layout, which lets older
    BCVAE checkpoints restore directly after moving the code into reinforcement_learning.residual_rl.
    """

    hil_serl_root: str = DEFAULT_HIL_SERL_ROOT
    pooling_method: str = "spatial_learned_embeddings"
    num_spatial_blocks: int = 8
    bottleneck_dim: int = 256
    num_image_views: int = 2

    def setup(self) -> None:
        if int(self.num_image_views) < 2:
            raise ValueError(f"num_image_views must be >= 2, got {self.num_image_views}")
        PreTrainedResNetEncoder, resnetv1_configs = _load_hil_serl_resnet_v1(
            self.hil_serl_root
        )

        self.encoder_img_main = PreTrainedResNetEncoder(
            pooling_method=self.pooling_method,
            num_spatial_blocks=self.num_spatial_blocks,
            bottleneck_dim=self.bottleneck_dim,
            pretrained_encoder=resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
            ),
            name="encoder_img_main",
        )
        self.encoder_img_extra = PreTrainedResNetEncoder(
            pooling_method=self.pooling_method,
            num_spatial_blocks=self.num_spatial_blocks,
            bottleneck_dim=self.bottleneck_dim,
            pretrained_encoder=resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
            ),
            name="encoder_img_extra",
        )
        self.extra_image_encoders = tuple(
            PreTrainedResNetEncoder(
                pooling_method=self.pooling_method,
                num_spatial_blocks=self.num_spatial_blocks,
                bottleneck_dim=self.bottleneck_dim,
                pretrained_encoder=resnetv1_configs["resnetv1-10-frozen"](
                    pre_pooling=True,
                ),
                name=f"encoder_img_extra_{view_idx}",
            )
            for view_idx in range(2, int(self.num_image_views))
        )

    @property
    def per_view_dim(self) -> int:
        return int(self.bottleneck_dim)

    def _encode_one(self, encoder: nn.Module, x: Array, train_backbone: bool) -> Array:
        # SERL encoders consume NHWC pixels in [0, 255] and apply their own
        # normalization internally, unlike the HF ResNet path above.
        x = jnp.transpose(x, (0, 2, 3, 1)) * 255.0
        return encoder(x, encode=True, train=train_backbone)

    def __call__(
        self,
        img_main: Array,
        img_extra: Array,
        *extra_images: Array,
        train_backbone: bool = False,
    ) -> Array:
        images = (img_main, img_extra) + tuple(extra_images)
        if len(images) != int(self.num_image_views):
            raise ValueError(
                f"HiLSerlResNet10Backbone expected {self.num_image_views} image "
                f"views, got {len(images)}."
            )
        feats = [
            self._encode_one(self.encoder_img_main, img_main, train_backbone),
            self._encode_one(self.encoder_img_extra, img_extra, train_backbone),
        ]
        feats.extend(
            self._encode_one(encoder, image, train_backbone)
            for encoder, image in zip(self.extra_image_encoders, extra_images)
        )
        return jnp.concatenate(feats, axis=-1)

    def encode_one(self, image: Array, train_backbone: bool = False) -> Array:
        """Encode a single NCHW image with the first SERL ResNet-10 branch."""
        return self._encode_one(self.encoder_img_main, image, train_backbone)


class BCVAEPolicy(nn.Module):
    """Single-step BC policy with a hand-action prior.

    Inputs:
        img_main/img_extra/extra_images: NCHW float images in `[0, 1]`.
        state: normalized 12D policy state, usually
            `concat(prev_arm_action_6, current_hand_abs_6)`.
        past_hand_win: absolute hand-pose history, shape `(batch, window, 6)`.
            VAE-prior and `pca_raw` checkpoints consume it raw. `mlp_direct`,
            `decoder_only`, and `vq_codebook` checkpoints consume the same
            history after hand-dim z-score normalization.

    Outputs:
        A dict containing the arm command, absolute hand action, latent hand
        control, and debug features.  Runtime adapters convert the absolute hand
        action to the environment's hand delta action.
    """

    vae_model_args: Dict[str, Any] | None = None
    backbone_type: str = "hf_resnet18"
    backbone_config: Dict[str, Any] | None = None
    state_encoder_type: str = "mlp"
    arm_state_dim: int = 6
    hand_state_dim: int = 6
    dropout: float = 0.0
    hand_prior_source: str = "vae"
    hand_codebook: Any | None = None
    hand_pca_mean: Any | None = None
    hand_pca_components: Any | None = None
    hand_pca_dim: int = 2
    window_size: int = 8
    hil_serl_root: str = DEFAULT_HIL_SERL_ROOT
    hil_serl_pooling_method: str = "spatial_learned_embeddings"
    hil_serl_num_spatial_blocks: int = 8
    hil_serl_bottleneck_dim: int = 256
    num_image_views: int = 2

    def setup(self) -> None:
        if int(self.num_image_views) < 2:
            raise ValueError(f"num_image_views must be >= 2, got {self.num_image_views}")
        # Hand-prior branch: VAE checkpoints are used by residual SAC and TD3
        # head finetuning, while direct/VQ variants are kept evaluable for BC.
        if self.hand_prior_source in {"vae", "decoder_only"}:
            if not self.vae_model_args:
                raise ValueError(
                    f"vae_model_args must be provided when hand_prior_source={self.hand_prior_source!r}"
                )
            self._latent_dim = int(self.vae_model_args["latent_dim"])
            if self.hand_prior_source == "decoder_only" and self._latent_dim != 2:
                raise ValueError(
                    f"decoder_only requires a 2D VAE latent, got {self._latent_dim}"
                )
            self.vae = HandActionVAE(name="vae", **self.vae_model_args)
        elif self.hand_prior_source == "mlp_direct":
            # Direct BC action head: no VAE latent bottleneck. A single
            # CoreActionHead predicts the full 12D next pose from visual,
            # encoded hand-history, and encoded arm-state features.
            self._latent_dim = 6
        elif self.hand_prior_source == "vq_codebook":
            if self.hand_codebook is None:
                raise ValueError(
                    "hand_codebook must be provided when hand_prior_source='vq_codebook'"
                )
            self._latent_dim = 1
        elif self.hand_prior_source == "pca_raw":
            if self.hand_pca_mean is None or self.hand_pca_components is None:
                raise ValueError(
                    "hand_pca_mean and hand_pca_components must be provided when hand_prior_source='pca_raw'"
                )
            self._latent_dim = int(self.hand_pca_dim)
            if self._latent_dim != 2:
                raise ValueError(f"pca_raw currently requires hand_pca_dim=2, got {self._latent_dim}")
        else:
            raise ValueError(f"Unknown hand_prior_source: {self.hand_prior_source}")

        # Visual branch: BC checkpoints may use either the HF ResNet-18 stack or
        # the SERL ResNet-10 stack used by the residual RL visual backbone.
        if self.backbone_type == "hf_resnet18":
            self.backbone = ResNet18Backbone(self.backbone_config, name="backbone")
            hidden_sizes = (self.backbone_config or {}).get(
                "hidden_sizes", [64, 128, 256, 512]
            )
            visual_feat_dim = int(hidden_sizes[-1]) * int(self.num_image_views)
        elif self.backbone_type == "hil_serl_resnet10":
            self.backbone = HiLSerlResNet10Backbone(
                hil_serl_root=self.hil_serl_root,
                pooling_method=self.hil_serl_pooling_method,
                num_spatial_blocks=self.hil_serl_num_spatial_blocks,
                bottleneck_dim=self.hil_serl_bottleneck_dim,
                num_image_views=int(self.num_image_views),
                name="backbone",
            )
            visual_feat_dim = int(self.num_image_views) * int(self.hil_serl_bottleneck_dim)
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        self.arm_state_encoder, arm_feat_dim = build_state_encoder_module(
            self.state_encoder_type
        )
        self.arm_feat_dim = self.arm_state_dim if arm_feat_dim is None else arm_feat_dim

        self.hand_prior_encoder, hand_prior_feat_dim = build_state_encoder_module(
            self.state_encoder_type
        )
        if self.hand_prior_source == "vae":
            hand_encoder_input_dim = 2 * self._latent_dim
        elif self.hand_prior_source == "pca_raw":
            hand_encoder_input_dim = int(self.window_size) * self._latent_dim
        else:
            hand_encoder_input_dim = int(self.window_size) * int(self.hand_state_dim)
        self.hand_feat_dim = hand_encoder_input_dim
        self.hand_prior_feat_dim = hand_encoder_input_dim if hand_prior_feat_dim is None else hand_prior_feat_dim

        if self.hand_prior_source == "mlp_direct":
            out_dim = self.arm_state_dim + self.hand_state_dim
        else:
            out_dim = self.arm_state_dim + self._latent_dim
        self.core_action_head = CoreActionHead(
            in_dim=visual_feat_dim + self.hand_prior_feat_dim + self.arm_feat_dim,
            out_dim=out_dim,
            dropout=self.dropout,
            name="core_action_head",
        )

    @property
    def latent_dim(self) -> int:
        """Return the active hand latent dimension for this checkpoint."""
        if self.hand_prior_source in {"vae", "decoder_only"}:
            return int(self.vae_model_args["latent_dim"])
        if self.hand_prior_source == "vq_codebook":
            return 1
        if self.hand_prior_source == "pca_raw":
            return int(self.hand_pca_dim)
        return 6

    def _pca_params(self, dtype) -> tuple[Array, Array]:
        mean = jnp.asarray(self.hand_pca_mean, dtype=dtype)
        components = jnp.asarray(self.hand_pca_components, dtype=dtype)[: int(self.hand_pca_dim)]
        return mean, components

    def encode_hand_pca_window(self, past_hand_win: Array) -> Array:
        mean, components = self._pca_params(past_hand_win.dtype)
        return jnp.matmul(past_hand_win - mean, jnp.swapaxes(components, -1, -2))

    def encode_visual(
        self,
        img_main: Array,
        img_extra: Array,
        train_backbone: bool,
        extra_images: tuple[Array, ...] = (),
    ) -> Array:
        """Encode the configured camera images and concatenate visual features."""
        extra_images = tuple(extra_images or ())
        num_inputs = 2 + len(extra_images)
        if num_inputs != int(self.num_image_views):
            raise ValueError(
                f"BCVAEPolicy expected {self.num_image_views} image views, "
                f"got {num_inputs}."
            )
        if self.backbone_type == "hil_serl_resnet10":
            return self.backbone(
                img_main,
                img_extra,
                *extra_images,
                train_backbone=train_backbone,
            )
        f_main = self.backbone(img_main, train_backbone=train_backbone)
        f_extra = self.backbone(img_extra, train_backbone=train_backbone)
        extra_feats = [
            self.backbone(image, train_backbone=train_backbone)
            for image in extra_images
        ]
        return jnp.concatenate([f_main, f_extra, *extra_feats], axis=-1)

    def encode_single_visual(self, image: Array, train_backbone: bool = False) -> Array:
        """Encode one BC-format NCHW image with the checkpoint's visual backbone."""
        if self.backbone_type == "hil_serl_resnet10":
            return self.backbone.encode_one(image, train_backbone=train_backbone)
        return self.backbone(image, train_backbone=train_backbone)

    def encode_hand_prior(self, past_hand_win: Array | None) -> tuple[Array, Array]:
        """Encode hand history into a latent prior `(mu, log_var)`."""
        if self.hand_prior_source == "vae":
            if past_hand_win is None:
                raise ValueError("past_hand_win is required when hand_prior_source='vae'")
            mu_p, lv_p = self.vae.encode(past_hand_win)
            # The VAE is a prior provider during BC/RL policy evaluation; policy
            # heads do not train through the VAE encoder.
            mu_p = lax.stop_gradient(mu_p)
            lv_p = lax.stop_gradient(lv_p)
            return mu_p, lv_p
        raise ValueError(f"hand_prior_source={self.hand_prior_source!r} has no latent hand prior")

    def decode_hand_latent(self, z: Array) -> Array:
        """Decode a latent hand code into an absolute hand pose."""
        if self.hand_prior_source in {"vae", "decoder_only"}:
            return self.vae.decode(z)
        if self.hand_prior_source == "pca_raw":
            mean, components = self._pca_params(z.dtype)
            return jnp.matmul(z, components) + mean
        raise ValueError(f"hand_prior_source={self.hand_prior_source!r} has no hand latent decoder")

    def decode_core_action(
        self,
        core_action: Array,
        reference_outputs: Dict[str, Array] | None = None,
    ) -> Dict[str, Array]:
        """Decode a semantic core action into BCVAE-style action outputs.

        `core_action` always starts with the 6D arm command.  The remaining
        dimensions use the checkpoint's native hand-core semantics: VAE delta-z,
        decoder/PCA latent, direct 6D hand pose, or VQ index.
        """
        arm_action = core_action[..., : self.arm_state_dim]
        hand_core = core_action[..., self.arm_state_dim :]

        if self.hand_prior_source == "vae":
            if reference_outputs is None or "mu_prior" not in reference_outputs:
                raise ValueError("reference_outputs['mu_prior'] is required for VAE core decoding")
            mu_p = reference_outputs["mu_prior"]
            lv_p = reference_outputs.get("log_var_prior", jnp.zeros_like(mu_p))
            delta_z = hand_core
            z_ctrl = mu_p + delta_z
            z_no_corr = mu_p
            hand_action = self.decode_hand_latent(z_ctrl)
            hand_no_corr = self.decode_hand_latent(z_no_corr)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_no_corr,
                "action_pred": jnp.concatenate([arm_action, hand_action], axis=-1),
                "mu_prior": mu_p,
                "log_var_prior": lv_p,
                "delta_z": delta_z,
                "core_action": core_action,
                "z_ctrl": z_ctrl,
                "z_no_corr": z_no_corr,
            }

        if self.hand_prior_source == "decoder_only":
            z_ctrl = hand_core
            z_no_corr = jnp.zeros_like(z_ctrl)
            hand_action = self.decode_hand_latent(z_ctrl)
            hand_no_corr = self.decode_hand_latent(z_no_corr)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_no_corr,
                "action_pred": jnp.concatenate([arm_action, hand_action], axis=-1),
                "core_action": core_action,
                "mu_prior": z_no_corr,
                "log_var_prior": z_no_corr,
                "delta_z": z_ctrl,
                "z_ctrl": z_ctrl,
                "z_no_corr": z_no_corr,
            }

        if self.hand_prior_source == "pca_raw":
            z_ctrl = hand_core
            zeros = jnp.zeros_like(z_ctrl)
            hand_action = self.decode_hand_latent(z_ctrl)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_action,
                "action_pred": jnp.concatenate([arm_action, hand_action], axis=-1),
                "core_action": core_action,
                "mu_prior": zeros,
                "log_var_prior": zeros,
                "delta_z": zeros,
                "z_ctrl": z_ctrl,
                "z_no_corr": zeros,
            }

        if self.hand_prior_source == "mlp_direct":
            hand_action = hand_core
            zeros = jnp.zeros_like(hand_action)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_action,
                "action_pred": jnp.concatenate([arm_action, hand_action], axis=-1),
                "core_action": core_action,
                "mu_prior": zeros,
                "log_var_prior": zeros,
                "delta_z": zeros,
                "z_ctrl": hand_action,
                "z_no_corr": zeros,
            }

        if self.hand_prior_source == "vq_codebook":
            hand_index_norm = hand_core
            codebook = jnp.asarray(self.hand_codebook, dtype=arm_action.dtype)
            index_float = (hand_index_norm[..., 0] + 1.0) * 0.5 * float(
                codebook.shape[0] - 1
            )
            hand_index = jnp.clip(
                jnp.floor(index_float), 0, codebook.shape[0] - 1
            ).astype(jnp.int32)
            hand_action = codebook[hand_index]
            zeros = jnp.zeros_like(hand_index_norm)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_action,
                "action_pred": jnp.concatenate([arm_action, hand_action], axis=-1),
                "core_action": core_action,
                "hand_index_norm": hand_index_norm,
                "hand_index": hand_index,
                "mu_prior": zeros,
                "log_var_prior": zeros,
                "delta_z": hand_index_norm,
                "z_ctrl": hand_index_norm,
                "z_no_corr": zeros,
            }

        raise ValueError(f"Unknown hand_prior_source: {self.hand_prior_source}")

    def __call__(
        self,
        img_main: Array,
        img_extra: Array,
        state: Array,
        past_hand_win: Array | None = None,
        extra_images: tuple[Array, ...] = (),
        zero_delta: bool = False,
        deterministic: bool = True,
        train_backbone: bool = False,
    ) -> Dict[str, Array]:
        """Run policy inference and return semantic action components."""
        visual_feat = self.encode_visual(
            img_main,
            img_extra,
            train_backbone=train_backbone,
            extra_images=extra_images,
        )
        arm_state = state[..., : self.arm_state_dim]
        arm_state_feat = self.arm_state_encoder(arm_state)

        if self.hand_prior_source in {"mlp_direct", "decoder_only", "vq_codebook"}:
            # Direct-input modes use the same state/history encoder structure as
            # the VAE path, but the hand encoder sees normalized hand history.
            if past_hand_win is None:
                raise ValueError(
                    f"past_hand_win is required when hand_prior_source={self.hand_prior_source!r}"
                )
            hand_state_chunk = jnp.reshape(
                past_hand_win,
                past_hand_win.shape[:-2] + (self.hand_feat_dim,),
            )
            hand_prior_feat = self.hand_prior_encoder(hand_state_chunk)
            core_action = self.core_action_head(
                jnp.concatenate([visual_feat, hand_prior_feat, arm_state_feat], axis=-1),
                deterministic=deterministic,
            )
            arm_action = core_action[..., : self.arm_state_dim]
            if self.hand_prior_source == "decoder_only":
                z_raw = core_action[
                    ..., self.arm_state_dim : self.arm_state_dim + self._latent_dim
                ]
                z_ctrl = jnp.zeros_like(z_raw) if zero_delta else z_raw
                z_no_corr = jnp.zeros_like(z_raw)
                hand_action = self.decode_hand_latent(z_ctrl)
                hand_no_corr = self.decode_hand_latent(z_no_corr)
                action_pred = jnp.concatenate([arm_action, hand_action], axis=-1)
                semantic_core_action = jnp.concatenate([arm_action, z_ctrl], axis=-1)
                return {
                    "arm_action": arm_action,
                    "hand_action": hand_action,
                    "hand_no_corr": hand_no_corr,
                    "action_pred": action_pred,
                    "core_action": semantic_core_action,
                    "mu_prior": z_no_corr,
                    "log_var_prior": z_no_corr,
                    "delta_z": z_ctrl,
                    "z_ctrl": z_ctrl,
                    "z_no_corr": z_no_corr,
                    "visual_feat": visual_feat,
                    "arm_state_feat": arm_state_feat,
                    "hand_prior_feat": hand_prior_feat,
                }
            if self.hand_prior_source == "vq_codebook":
                hand_index_norm = core_action[
                    ..., self.arm_state_dim : self.arm_state_dim + 1
                ]
                codebook = jnp.asarray(self.hand_codebook, dtype=arm_action.dtype)
                index_float = (hand_index_norm[..., 0] + 1.0) * 0.5 * float(
                    codebook.shape[0] - 1
                )
                hand_index = jnp.clip(
                    jnp.floor(index_float), 0, codebook.shape[0] - 1
                ).astype(jnp.int32)
                hand_action = codebook[hand_index]
                action_pred = jnp.concatenate([arm_action, hand_action], axis=-1)
                zeros = jnp.zeros_like(hand_index_norm)
                return {
                    "arm_action": arm_action,
                    "hand_action": hand_action,
                    "hand_no_corr": hand_action,
                    "action_pred": action_pred,
                    "core_action": core_action,
                    "hand_index_norm": hand_index_norm,
                    "hand_index": hand_index,
                    "mu_prior": zeros,
                    "log_var_prior": zeros,
                    "delta_z": hand_index_norm,
                    "z_ctrl": hand_index_norm,
                    "z_no_corr": zeros,
                    "visual_feat": visual_feat,
                    "arm_state_feat": arm_state_feat,
                    "hand_prior_feat": hand_prior_feat,
                }
            hand_action = core_action[
                ..., self.arm_state_dim : self.arm_state_dim + self.hand_state_dim
            ]
            action_pred = jnp.concatenate([arm_action, hand_action], axis=-1)
            zeros = jnp.zeros_like(hand_action)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_action,
                "action_pred": action_pred,
                "core_action": core_action,
                "mu_prior": zeros,
                "log_var_prior": zeros,
                "delta_z": zeros,
                "z_ctrl": hand_action,
                "z_no_corr": zeros,
                "visual_feat": visual_feat,
                "arm_state_feat": arm_state_feat,
                "hand_prior_feat": hand_prior_feat,
            }

        if self.hand_prior_source == "pca_raw":
            if past_hand_win is None:
                raise ValueError("past_hand_win is required when hand_prior_source='pca_raw'")
            pca_window = self.encode_hand_pca_window(past_hand_win)
            hand_state_chunk = jnp.reshape(
                pca_window,
                pca_window.shape[:-2] + (int(self.window_size) * self._latent_dim,),
            )
            hand_prior_feat = self.hand_prior_encoder(hand_state_chunk)
            hand_head_input = jnp.concatenate([visual_feat, hand_prior_feat, arm_state_feat], axis=-1)
            core_action = self.core_action_head(
                hand_head_input,
                deterministic=deterministic,
            )
            arm_action = core_action[..., : self.arm_state_dim]
            z_ctrl = core_action[
                ..., self.arm_state_dim : self.arm_state_dim + self._latent_dim
            ]
            hand_action = self.decode_hand_latent(z_ctrl)
            action_pred = jnp.concatenate([arm_action, hand_action], axis=-1)
            zeros = jnp.zeros_like(z_ctrl)
            return {
                "arm_action": arm_action,
                "hand_action": hand_action,
                "hand_no_corr": hand_action,
                "action_pred": action_pred,
                "mu_prior": zeros,
                "log_var_prior": zeros,
                "delta_z": zeros,
                "core_action": core_action,
                "z_ctrl": z_ctrl,
                "z_no_corr": zeros,
                "visual_feat": visual_feat,
                "arm_state_feat": arm_state_feat,
                "hand_prior_feat": hand_prior_feat,
            }

        mu_p, lv_p = self.encode_hand_prior(past_hand_win)
        hand_prior_feat = self.hand_prior_encoder(jnp.concatenate([mu_p, lv_p], axis=-1))
        hand_head_input = jnp.concatenate([visual_feat, hand_prior_feat, arm_state_feat], axis=-1)

        core_action_raw = self.core_action_head(
            hand_head_input,
            deterministic=deterministic,
        )
        arm_action = core_action_raw[..., : self.arm_state_dim]
        delta_z_raw = core_action_raw[
            ..., self.arm_state_dim : self.arm_state_dim + self._latent_dim
        ]
        if zero_delta:
            delta_z = jnp.zeros_like(delta_z_raw)
        else:
            delta_z = delta_z_raw
        core_action = jnp.concatenate([arm_action, delta_z], axis=-1)

        z_ctrl = mu_p + delta_z
        z_no_corr = mu_p
        hand_action = self.decode_hand_latent(z_ctrl)
        hand_no_corr = self.decode_hand_latent(z_no_corr)
        action_pred = jnp.concatenate([arm_action, hand_action], axis=-1)

        return {
            "arm_action": arm_action,
            "hand_action": hand_action,
            "hand_no_corr": hand_no_corr,
            "action_pred": action_pred,
            "mu_prior": mu_p,
            "log_var_prior": lv_p,
            "delta_z": delta_z,
            "core_action": core_action,
            "z_ctrl": z_ctrl,
            "z_no_corr": z_no_corr,
            "visual_feat": visual_feat,
            "arm_state_feat": arm_state_feat,
            "hand_prior_feat": hand_prior_feat,
        }


def count_params(tree: Dict[str, Any]) -> int:
    """Return the total number of scalar parameters in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(leaf.size for leaf in leaves))

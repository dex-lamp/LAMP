"""Residual SAC agent layered on top of a frozen BCVAE policy.

The actor network learns a residual in the frozen BC policy's semantic
`core_action` space.  Environment interaction and critic training still use the
repository's standard 12D robot action.  This module owns that
residual-to-env-action conversion while delegating BCVAE checkpoint
loading/preprocessing to `reinforcement_learning.residual_rl.utils.bc_vae` and online context creation to
`BCVAEContextWrapper`.
"""

import copy
from functools import partial
from typing import Any, Iterable, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from reinforcement_learning.residual_rl.agents.continuous.sac import SACAgent
from reinforcement_learning.residual_rl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from reinforcement_learning.residual_rl.common.optimizers import make_optimizer
from reinforcement_learning.residual_rl.common.typing import Batch, Data, Params, PRNGKey
from reinforcement_learning.residual_rl.networks.actor_critic_nets import Critic, Policy, ensemblize
from reinforcement_learning.residual_rl.networks.bc_vae import (
    DEFAULT_HIL_SERL_ROOT,
    ResNet18Backbone,
    _load_hil_serl_resnet_v1,
)
from reinforcement_learning.residual_rl.networks.lagrange import GeqLagrangeMultiplier
from reinforcement_learning.residual_rl.networks.mlp import MLP
from reinforcement_learning.residual_rl.utils.bc_vae import prepare_bc_vae_image, prepare_bc_vae_inputs
from reinforcement_learning.residual_rl.utils.train_utils import _unpack


def _bc_core_feature_inputs_from_outputs(bc_out: Data, batched: bool) -> Data:
    """Build trainable head inputs from frozen BCVAE core-action-head signals."""
    required = ("visual_feat", "hand_prior_feat", "arm_state_feat", "core_action")
    missing = [key for key in required if key not in bc_out]
    if missing:
        raise KeyError(f"BCVAE output missing residual actor inputs: {missing}")
    inputs = {key: bc_out[key] for key in required}
    if not batched:
        inputs = {key: value[0] for key, value in inputs.items()}
    return inputs


def _residual_feature_inputs_from_outputs(
    bc_out: Data,
    observations: Data,
    batched: bool,
    residual_extra_image_keys: Iterable[str] = (),
) -> Data:
    """Build residual actor/critic inputs from BCVAE features plus extra views."""
    inputs = _bc_core_feature_inputs_from_outputs(bc_out, batched)
    for image_key in tuple(residual_extra_image_keys or ()):
        if image_key not in observations:
            raise KeyError(
                f"Residual extra image key {image_key!r} is missing from observations."
            )
        inputs[image_key] = observations[image_key]
    return inputs


def _make_residual_feature_inputs(
    *,
    bc_model: nn.Module,
    bc_variables: flax.core.FrozenDict,
    observations: Data,
    action_mean: jax.Array,
    action_std: jax.Array,
    bc_image_keys: Iterable[str],
    residual_extra_image_keys: Iterable[str],
    normalize_hand_history: bool,
    hand_history_std_floor: float,
    zero_delta: bool,
) -> Data:
    img_main, img_extra, state, past_hand_win, _, obs_batched = prepare_bc_vae_inputs(
        observations,
        action_mean=action_mean,
        action_std=action_std,
        image_keys=bc_image_keys,
        normalize_hand_history=normalize_hand_history,
        hand_history_std_floor=hand_history_std_floor,
    )
    bc_out = bc_model.apply(
        bc_variables,
        img_main,
        img_extra,
        state,
        past_hand_win,
        zero_delta=zero_delta,
        deterministic=True,
        train_backbone=False,
    )
    return _residual_feature_inputs_from_outputs(
        bc_out,
        observations,
        obs_batched,
        residual_extra_image_keys=residual_extra_image_keys,
    )


class SingleHiLSerlResNet10Backbone(nn.Module):
    """Single-camera version of the BCVAE SERL ResNet-10 visual backbone."""

    hil_serl_root: str = DEFAULT_HIL_SERL_ROOT
    pooling_method: str = "spatial_learned_embeddings"
    num_spatial_blocks: int = 8
    bottleneck_dim: int = 256

    def setup(self) -> None:
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

    def __call__(self, image: jax.Array, train_backbone: bool = False) -> jax.Array:
        image = jnp.transpose(image, (0, 2, 3, 1)) * 255.0
        return self.encoder_img_main(image, encode=True, train=train_backbone)


class BCAlignedExtraVisualEncoder(nn.Module):
    """Encode residual-only views with the same backbone family as BCVAE."""

    backbone_type: str
    backbone_config: Optional[dict[str, Any]] = None
    hil_serl_root: str = DEFAULT_HIL_SERL_ROOT
    hil_serl_pooling_method: str = "spatial_learned_embeddings"
    hil_serl_num_spatial_blocks: int = 8
    hil_serl_bottleneck_dim: int = 256

    def setup(self) -> None:
        if self.backbone_type == "hf_resnet18":
            self.backbone = ResNet18Backbone(
                self.backbone_config,
                name="backbone",
            )
        elif self.backbone_type == "hil_serl_resnet10":
            self.backbone = SingleHiLSerlResNet10Backbone(
                hil_serl_root=self.hil_serl_root,
                pooling_method=self.hil_serl_pooling_method,
                num_spatial_blocks=self.hil_serl_num_spatial_blocks,
                bottleneck_dim=self.hil_serl_bottleneck_dim,
                name="backbone",
            )
        else:
            raise ValueError(f"Unknown BCVAE backbone_type: {self.backbone_type}")

    def __call__(
        self,
        image: jax.Array,
        train: bool = False,
        encode: bool = True,
        batched: Optional[bool] = None,
    ) -> jax.Array:
        del train, encode
        if batched is None:
            batched = image.ndim == 5
        image = prepare_bc_vae_image(image, batched=batched)
        feat = self.backbone(image, train_backbone=False)
        return feat if batched else feat[0]


def _make_bc_aligned_extra_visual_encoders(
    image_keys: Iterable[str],
    *,
    backbone_type: str,
    backbone_config: Optional[dict[str, Any]] = None,
    hil_serl_root: str = DEFAULT_HIL_SERL_ROOT,
    hil_serl_pooling_method: str = "spatial_learned_embeddings",
    hil_serl_num_spatial_blocks: int = 8,
    hil_serl_bottleneck_dim: int = 256,
) -> dict[str, nn.Module]:
    return {
        image_key: BCAlignedExtraVisualEncoder(
            backbone_type=backbone_type,
            backbone_config=backbone_config,
            hil_serl_root=hil_serl_root,
            hil_serl_pooling_method=hil_serl_pooling_method,
            hil_serl_num_spatial_blocks=hil_serl_num_spatial_blocks,
            hil_serl_bottleneck_dim=hil_serl_bottleneck_dim,
            name=f"encoder_{image_key}",
        )
        for image_key in tuple(image_keys or ())
    }


def _make_extra_visual_encoders(
    encoder_type: str,
    image_keys: Iterable[str],
) -> dict[str, nn.Module]:
    """Build SERL-style per-camera encoders for residual-only extra views."""
    image_keys = tuple(image_keys or ())
    if not image_keys:
        return {}

    if encoder_type == "resnet":
        from reinforcement_learning.residual_rl.vision.resnet_v1 import resnetv1_configs

        return {
            image_key: resnetv1_configs["resnetv1-10"](
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    if encoder_type == "resnet-pretrained":
        from reinforcement_learning.residual_rl.vision.resnet_v1 import (
            PreTrainedResNetEncoder,
            resnetv1_configs,
        )

        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True,
            name="pretrained_encoder",
        )
        return {
            image_key: PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    raise NotImplementedError(f"Unknown encoder type: {encoder_type}")


def _mutable_copy_tree(tree):
    if tree is None:
        return None
    if isinstance(tree, flax.core.FrozenDict):
        return flax.core.unfreeze(tree)
    if isinstance(tree, dict):
        return {key: _mutable_copy_tree(value) for key, value in tree.items()}
    return tree


def _restore_tree_container(tree, reference_tree):
    if reference_tree is None:
        return tree
    if isinstance(reference_tree, flax.core.FrozenDict):
        return flax.core.freeze(tree)
    return tree


def _has_tree(tree) -> bool:
    return tree is not None and len(tree) > 0


def _restore_extra_encoder_param_subtrees(
    updated_tree,
    reference_tree,
    image_keys: Iterable[str],
):
    """Copy frozen extra encoder params from reference into an updated tree."""
    image_keys = tuple(image_keys or ())
    if not image_keys:
        return updated_tree
    updated = _mutable_copy_tree(updated_tree)
    reference = _mutable_copy_tree(reference_tree)
    for module_name in ("modules_actor", "modules_critic"):
        updated_encoder = updated.get(module_name, {}).get("encoder", {})
        reference_encoder = reference.get(module_name, {}).get("encoder", {})
        for image_key in image_keys:
            encoder_name = f"encoder_{image_key}"
            if encoder_name in updated_encoder and encoder_name in reference_encoder:
                updated_encoder[encoder_name] = reference_encoder[encoder_name]
    return _restore_tree_container(updated, updated_tree)


def _load_residual_extra_resnet10_params(
    agent,
    image_keys: Iterable[str],
    params_path: str = "pretrained_models/resnet10_params.pkl",
):
    """Load local ImageNet ResNet-10 weights into extra-view encoders."""
    image_keys = tuple(image_keys or ())
    if not image_keys:
        return agent

    import pickle as pkl
    from pathlib import Path

    params_path = Path(params_path).expanduser()
    if not params_path.exists():
        raise FileNotFoundError(
            f"ResNet-10 params not found at {params_path}. "
            "Provide local weights before using residual_extra_encoder_type='serl'."
        )
    with params_path.open("rb") as f:
        encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params))
    print(f"Loaded {param_count / 1e6:.1f}M ResNet-10 parameters from {params_path}")

    original_params = agent.state.params
    original_target_params = agent.state.target_params
    params = _mutable_copy_tree(original_params)
    target_params = _mutable_copy_tree(original_target_params)
    for param_tree in (params, target_params):
        for module_name in ("modules_actor", "modules_critic"):
            encoder_tree = param_tree.get(module_name, {}).get("encoder", {})
            for image_key in image_keys:
                encoder_name = f"encoder_{image_key}"
                if encoder_name not in encoder_tree:
                    raise KeyError(
                        f"Could not find residual extra encoder params "
                        f"{module_name}/encoder/{encoder_name}"
                    )
                new_encoder_params = encoder_tree[encoder_name]
                if "pretrained_encoder" in new_encoder_params:
                    new_encoder_params = new_encoder_params["pretrained_encoder"]
                for key in new_encoder_params:
                    if key in encoder_params:
                        new_encoder_params[key] = encoder_params[key]
                        print(f"replaced {key} in {module_name}/{encoder_name}")

    return agent.replace(
        state=agent.state.replace(
            params=_restore_tree_container(params, original_params),
            target_params=_restore_tree_container(target_params, original_target_params),
        )
    )


def _load_residual_extra_bc_backbone_params(
    agent,
    image_keys: Iterable[str],
    *,
    backbone_type: str,
    bc_backbone_params,
    bc_backbone_batch_stats=None,
):
    """Initialize residual extra-view encoders from the frozen BCVAE backbone."""
    image_keys = tuple(image_keys or ())
    if not image_keys:
        return agent
    if bc_backbone_params is None:
        raise ValueError("BC-aligned residual extra encoders require BC backbone params.")

    original_params = agent.state.params
    original_target_params = agent.state.target_params
    original_batch_stats = agent.batch_stats
    params = _mutable_copy_tree(original_params)
    target_params = _mutable_copy_tree(original_target_params)
    batch_stats = _mutable_copy_tree(original_batch_stats)
    source_params = _mutable_copy_tree(bc_backbone_params)
    source_batch_stats = _mutable_copy_tree(bc_backbone_batch_stats)

    if backbone_type == "hil_serl_resnet10":
        source_params = {"encoder_img_main": source_params["encoder_img_main"]}
        if source_batch_stats and "encoder_img_main" in source_batch_stats:
            source_batch_stats = {
                "encoder_img_main": source_batch_stats["encoder_img_main"]
            }
        else:
            source_batch_stats = None

    for param_tree in (params, target_params):
        for module_name in ("modules_actor", "modules_critic"):
            encoder_tree = param_tree.get(module_name, {}).get("encoder", {})
            for image_key in image_keys:
                encoder_name = f"encoder_{image_key}"
                if encoder_name not in encoder_tree:
                    raise KeyError(
                        f"Could not find residual extra encoder params "
                        f"{module_name}/encoder/{encoder_name}"
                    )
                encoder_tree[encoder_name]["backbone"] = copy.deepcopy(source_params)
                print(
                    f"initialized {module_name}/{encoder_name} from "
                    f"BCVAE {backbone_type} backbone"
                )

    if batch_stats is not None and source_batch_stats:
        for module_name in ("modules_actor", "modules_critic"):
            encoder_tree = batch_stats.get(module_name, {}).get("encoder", {})
            for image_key in image_keys:
                encoder_name = f"encoder_{image_key}"
                if encoder_name in encoder_tree:
                    encoder_tree[encoder_name]["backbone"] = copy.deepcopy(
                        source_batch_stats
                    )

    return agent.replace(
        state=agent.state.replace(
            params=_restore_tree_container(params, original_params),
            target_params=_restore_tree_container(target_params, original_target_params),
        ),
        batch_stats=_restore_tree_container(batch_stats, original_batch_stats),
    )


class BCCoreActionFeatureEncoder(nn.Module):
    """Encode BCVAE core-head features for the residual actor.

    The frozen BCVAE supplies the same tensors used by its `core_action_head`:
    visual, hand-history/prior, and arm-state features.  The BC core action gets
    a small trainable projection before all four feature groups are concatenated
    and passed to the policy MLP.
    """

    core_action_embed_dim: int = 64
    residual_extra_image_keys: Tuple[str, ...] = ()
    encoder: Optional[dict[str, nn.Module]] = None
    enable_stacking: bool = True
    freeze_extra_encoders: bool = True

    @nn.compact
    def __call__(
        self,
        observations: Data,
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
        del stop_gradient
        visual_feat = jax.lax.stop_gradient(jnp.asarray(observations["visual_feat"]))
        hand_prior_feat = jax.lax.stop_gradient(
            jnp.asarray(observations["hand_prior_feat"])
        )
        arm_state_feat = jax.lax.stop_gradient(jnp.asarray(observations["arm_state_feat"]))
        core_action = jax.lax.stop_gradient(jnp.asarray(observations["core_action"]))

        core_action_feat = nn.Dense(
            self.core_action_embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
        )(core_action)
        core_action_feat = nn.LayerNorm()(core_action_feat)
        core_action_feat = nn.tanh(core_action_feat)

        encoded = [visual_feat, hand_prior_feat, arm_state_feat, core_action_feat]
        extra_encoders = self.encoder or {}
        for image_key in self.residual_extra_image_keys:
            if image_key not in extra_encoders:
                raise KeyError(f"No residual extra encoder configured for {image_key!r}")
            image = observations[image_key]
            if not is_encoded and self.enable_stacking:
                if len(image.shape) == 4:
                    image = rearrange(image, "T H W C -> H W (T C)")
                elif len(image.shape) == 5:
                    image = rearrange(image, "B T H W C -> B H W (T C)")
            encoder_kwargs = {
                "train": train,
                "encode": not is_encoded,
            }
            if not self.enable_stacking:
                encoder_kwargs["batched"] = visual_feat.ndim >= 2
            image_feat = extra_encoders[image_key](image, **encoder_kwargs)
            if self.freeze_extra_encoders:
                image_feat = jax.lax.stop_gradient(image_feat)
            encoded.append(image_feat)

        return jnp.concatenate(encoded, axis=-1)


class ResidualSACAgent(SACAgent):
    """SAC agent whose policy samples a core-action residual over frozen BC.

    The actor distribution lives in the BC checkpoint's semantic core-action
    space:
      [6-D arm residual, N-D hand-core residual].

    The critic still receives the 12-D environment action:
      [6-D arm command, 6-D hand delta].
    """

    bc_model: nn.Module = nonpytree_field()
    bc_variables: flax.core.FrozenDict
    action_mean: jax.Array
    action_std: jax.Array
    action_low: jax.Array
    action_high: jax.Array
    batch_stats: Optional[flax.core.FrozenDict] = None

    def _prepare_bc_inputs(self, observations: Data):
        """Prepare augmented SERL observations for the frozen BCVAE model.

        Residual SAC learns in residual space, but every actor/target action has
        to pass through BCVAE first.  This shared utility keeps image layout,
        state normalization, and context batching identical to eval and TD3.
        """
        return prepare_bc_vae_inputs(
            observations,
            action_mean=self.action_mean,
            action_std=self.action_std,
            image_keys=self.config["bc_image_keys"],
            normalize_hand_history=self.config["normalize_hand_history"],
            hand_history_std_floor=self.config.get("hand_history_std_floor", 1e-6),
        )

    def _forward_bc_and_context(self, observations: Data):
        img_main, img_extra, state, past_hand_win, current_hand_abs, obs_batched = (
            self._prepare_bc_inputs(observations)
        )
        bc_out = self.bc_model.apply(
            self.bc_variables,
            img_main,
            img_extra,
            state,
            past_hand_win,
            zero_delta=self.config["zero_delta"],
            deterministic=True,
            train_backbone=False,
        )
        return bc_out, current_hand_abs, obs_batched

    def forward_bc(self, observations: Data):
        bc_out, _, _ = self._forward_bc_and_context(observations)
        return bc_out

    def _network_variables(self, params: Params) -> dict[str, Any]:
        variables = {"params": params}
        if _has_tree(self.batch_stats):
            variables["batch_stats"] = self.batch_stats
        return variables

    def _forward_policy_from_bc(
        self,
        bc_out: Data,
        observations: Data,
        obs_batched: bool,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        if train:
            assert rng is not None, "Must specify rng when training"
        actor_inputs = _residual_feature_inputs_from_outputs(
            bc_out,
            observations,
            obs_batched,
            residual_extra_image_keys=self.config["residual_extra_image_keys"],
        )
        return self.state.apply_fn(
            self._network_variables(grad_params or self.state.params),
            actor_inputs,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def _forward_critic_from_bc(
        self,
        bc_out: Data,
        observations: Data,
        obs_batched: bool,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        if train:
            assert rng is not None, "Must specify rng when training"
        critic_inputs = _residual_feature_inputs_from_outputs(
            bc_out,
            observations,
            obs_batched,
            residual_extra_image_keys=self.config["residual_extra_image_keys"],
        )
        return self.state.apply_fn(
            self._network_variables(grad_params or self.state.params),
            critic_inputs,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        bc_out, _, obs_batched = self._forward_bc_and_context(observations)
        return self._forward_policy_from_bc(
            bc_out,
            observations,
            obs_batched,
            rng=rng,
            grad_params=grad_params,
            train=train,
        )

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        bc_out, _, obs_batched = self._forward_bc_and_context(observations)
        return self._forward_critic_from_bc(
            bc_out,
            observations,
            obs_batched,
            actions,
            rng,
            grad_params=grad_params,
            train=train,
        )

    def decode_hand_latent(self, z: jax.Array) -> jax.Array:
        """Decode a hand latent through the frozen BCVAE decoder."""
        return self.bc_model.apply(
            self.bc_variables,
            z,
            method=self.bc_model.decode_hand_latent,
        )

    def decode_core_action(self, core_action: jax.Array, reference_outputs: Data):
        """Decode adjusted core action through the frozen BC action adapter."""
        return self.bc_model.apply(
            self.bc_variables,
            core_action,
            reference_outputs,
            method=self.bc_model.decode_core_action,
        )

    def _residual_scale_vector(self, dtype) -> jax.Array:
        return jnp.concatenate(
            [
                jnp.full((6,), self.config["residual_scale_arm"], dtype=dtype),
                jnp.full(
                    (self.config["hand_core_dim"],),
                    self.config["residual_scale_core"],
                    dtype=dtype,
                ),
            ],
            axis=0,
        )

    def residuals_to_env_actions(
        self,
        observations: Data,
        residual_actions: jax.Array,
        *,
        clip: bool,
        reference_outputs: Optional[Data] = None,
        current_hand_abs: Optional[jax.Array] = None,
        obs_batched: Optional[bool] = None,
    ) -> jax.Array:
        """Convert residual actor output to the 12D env action.

        The residual policy outputs `[arm_residual_6, hand_core_residual_N]`.
        This function evaluates frozen BCVAE on the same observation, applies
        configured residual scales in core-action space, decodes the adjusted
        hand core to absolute hand pose, then subtracts `current_hand_abs` to
        obtain the environment's hand delta.
        """
        residual_actions = jnp.asarray(residual_actions)
        batched = residual_actions.ndim == 2
        residual_actions_batched = residual_actions if batched else residual_actions[None, ...]

        if reference_outputs is None or current_hand_abs is None or obs_batched is None:
            reference_outputs, current_hand_abs, obs_batched = self._forward_bc_and_context(
                observations
            )
        if obs_batched != batched:
            raise ValueError(
                "Observation batch rank and residual action batch rank do not match: "
                f"obs_batched={obs_batched}, residual_batched={batched}"
            )

        core_bc = jax.lax.stop_gradient(reference_outputs["core_action"])
        chex.assert_equal_shape([core_bc, residual_actions_batched])

        scale = self._residual_scale_vector(core_bc.dtype)
        core_cmd = core_bc + scale * residual_actions_batched
        residual_out = self.decode_core_action(core_cmd, reference_outputs)
        arm_cmd = residual_out["arm_action"]
        hand_delta = residual_out["hand_action"] - current_hand_abs

        env_actions = jnp.concatenate([arm_cmd, hand_delta], axis=-1)
        if clip:
            env_actions = jnp.clip(env_actions, self.action_low, self.action_high)
        return env_actions if batched else env_actions[0]

    def _compute_next_actions(self, batch, rng):
        batch_size = batch["rewards"].shape[0]

        next_bc_out, next_current_hand_abs, next_obs_batched = (
            self._forward_bc_and_context(batch["next_observations"])
        )
        next_action_distributions = self._forward_policy_from_bc(
            next_bc_out,
            batch["next_observations"],
            next_obs_batched,
            rng=rng,
        )
        next_residuals, next_residual_log_probs = (
            next_action_distributions.sample_and_log_prob(seed=rng)
        )
        chex.assert_shape(next_residuals, (batch_size, self.config["residual_action_dim"]))
        chex.assert_shape(next_residual_log_probs, (batch_size,))

        next_actions = self.residuals_to_env_actions(
            batch["next_observations"],
            next_residuals,
            clip=False,
            reference_outputs=next_bc_out,
            current_hand_abs=next_current_hand_abs,
            obs_batched=next_obs_batched,
        )
        chex.assert_equal_shape([batch["actions"], next_actions])
        return next_actions, next_residual_log_probs

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        bc_out, current_hand_abs, obs_batched = self._forward_bc_and_context(
            batch["observations"]
        )
        action_distributions = self._forward_policy_from_bc(
            bc_out,
            batch["observations"],
            obs_batched,
            rng=policy_rng,
            grad_params=params,
        )
        residuals, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)
        actions = self.residuals_to_env_actions(
            batch["observations"],
            residuals,
            clip=False,
            reference_outputs=bc_out,
            current_hand_abs=current_hand_abs,
            obs_batched=obs_batched,
        )

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "residual_l2": jnp.linalg.norm(residuals, axis=-1).mean(),
            "adapted_action_l2": jnp.linalg.norm(actions, axis=-1).mean(),
        }

        return actor_loss, info

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update=frozenset({"actor", "critic", "temperature"}),
        **kwargs,
    ) -> Tuple["ResidualSACAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if (
            "augmentation_function" in self.config
            and self.config["augmentation_function"] is not None
        ):
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        loss_fns = self.loss_fns(batch, **kwargs)
        assert networks_to_update.issubset(loss_fns.keys()), (
            f"Invalid gradient steps: {networks_to_update}"
        )
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        if self.config.get("freeze_residual_extra_encoders", False):
            frozen_extra_keys = self.config["residual_extra_image_keys"]
            new_state = new_state.replace(
                params=_restore_extra_encoder_param_subtrees(
                    new_state.params,
                    self.state.params,
                    frozen_extra_keys,
                ),
                target_params=_restore_extra_encoder_param_subtrees(
                    new_state.target_params,
                    self.state.target_params,
                    frozen_extra_keys,
                ),
            )

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """Sample an environment action for actor rollout or checkpoint eval."""
        bc_out, current_hand_abs, obs_batched = self._forward_bc_and_context(observations)
        dist = self._forward_policy_from_bc(
            bc_out,
            observations,
            obs_batched,
            rng=seed,
            train=False,
        )
        residuals = dist.mode() if argmax else dist.sample(seed=seed)
        return self.residuals_to_env_actions(
            observations,
            residuals,
            clip=True,
            reference_outputs=bc_out,
            current_hand_abs=current_hand_abs,
            obs_batched=obs_batched,
        )

    def debug_action_outputs(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Return residual, BC core, adapted core, and env-action outputs."""
        del kwargs
        if seed is None:
            seed = jax.random.PRNGKey(0)
        return self._debug_action_outputs(observations, seed, argmax=argmax)

    @partial(jax.jit, static_argnames=("argmax",))
    def _debug_action_outputs(
        self,
        observations: Data,
        seed: PRNGKey,
        *,
        argmax: bool = False,
    ) -> dict[str, jax.Array]:
        bc_out, current_hand_abs, obs_batched = self._forward_bc_and_context(
            observations
        )
        dist = self._forward_policy_from_bc(
            bc_out,
            observations,
            obs_batched,
            rng=seed,
            train=False,
        )
        residuals = dist.mode() if argmax else dist.sample(seed=seed)
        raw_env_action = self.residuals_to_env_actions(
            observations,
            residuals,
            clip=False,
            reference_outputs=bc_out,
            current_hand_abs=current_hand_abs,
            obs_batched=obs_batched,
        )
        sent_env_action = self.residuals_to_env_actions(
            observations,
            residuals,
            clip=True,
            reference_outputs=bc_out,
            current_hand_abs=current_hand_abs,
            obs_batched=obs_batched,
        )

        bc_core_action = jax.lax.stop_gradient(bc_out["core_action"])
        residuals_batched = residuals if residuals.ndim == 2 else residuals[None, ...]
        scale = self._residual_scale_vector(bc_core_action.dtype)
        scaled_residual = scale * residuals_batched
        adapted_core_action = bc_core_action + scaled_residual
        adapted_outputs = self.decode_core_action(adapted_core_action, bc_out)

        def maybe_unbatch(value):
            return value if obs_batched else value[0]

        debug = {
            "sent_env_action": sent_env_action,
            "raw_env_action": raw_env_action,
            "sent_arm_action": sent_env_action[..., :6],
            "raw_arm_action": raw_env_action[..., :6],
            "sent_hand_delta": sent_env_action[..., 6:12],
            "raw_hand_delta": raw_env_action[..., 6:12],
            "residual_head_output": residuals,
            "residual_action": residuals,
            "bc_core_action": maybe_unbatch(bc_core_action),
            "core_action_head_output": maybe_unbatch(bc_core_action),
            "core_action": maybe_unbatch(bc_core_action),
            "scaled_residual_action": maybe_unbatch(scaled_residual),
            "adapted_core_action": maybe_unbatch(adapted_core_action),
            "residual_scale_vector": scale,
            "current_hand_abs": maybe_unbatch(current_hand_abs),
            "bc_policy_state": jnp.asarray(observations["bc_policy_state"]),
        }

        if "past_hand_win" in observations:
            debug["past_hand_win"] = jnp.asarray(observations["past_hand_win"])

        for key in (
            "arm_action",
            "delta_z",
            "z_ctrl",
            "z_no_corr",
            "hand_action",
            "hand_no_corr",
            "mu_prior",
            "log_var_prior",
            "action_pred",
            "hand_index",
            "hand_index_norm",
            "visual_feat",
            "arm_state_feat",
            "hand_prior_feat",
        ):
            if key in bc_out:
                debug[f"bc_{key}"] = maybe_unbatch(bc_out[key])
            if key in adapted_outputs:
                debug[f"adapted_{key}"] = maybe_unbatch(adapted_outputs[key])

        return debug

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        env_actions: jnp.ndarray,
        *,
        bc_model: nn.Module,
        bc_variables: flax.core.FrozenDict,
        action_mean: jax.Array,
        action_std: jax.Array,
        action_low: jax.Array,
        action_high: jax.Array,
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        actor_observations: Optional[Data] = None,
        critic_observations: Optional[Data] = None,
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        bc_image_keys: Iterable[str] = ("global", "wrist"),
        residual_extra_image_keys: Iterable[str] = (),
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        residual_action_dim: Optional[int] = None,
        hand_prior_source: str = "vae",
        hand_core_dim: int = 2,
        core_action_dim: int = 8,
        residual_scale_arm: float = 0.05,
        residual_scale_core: float = 0.05,
        zero_delta: bool = False,
        normalize_hand_history: bool = False,
        **kwargs,
    ):
        """Create residual SAC train state plus frozen BCVAE runtime fields."""
        if residual_action_dim is None:
            residual_action_dim = int(core_action_dim)
        if int(residual_action_dim) != int(core_action_dim):
            raise ValueError(
                "Residual action dim must match BC core action dim: "
                f"residual_action_dim={residual_action_dim}, "
                f"core_action_dim={core_action_dim}"
            )

        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        actor_init_observations = (
            observations if actor_observations is None else actor_observations
        )
        critic_init_observations = (
            observations if critic_observations is None else critic_observations
        )
        variables = model_def.init(
            init_rng,
            actor=[actor_init_observations],
            critic=[critic_init_observations, env_actions],
            temperature=[],
        )
        params = variables["params"]
        batch_stats = variables.get("batch_stats", None)

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if target_entropy is None:
            target_entropy = int(residual_action_dim) / 2

        return cls(
            state=state,
            bc_model=bc_model,
            bc_variables=bc_variables,
            action_mean=jnp.asarray(action_mean, dtype=jnp.float32),
            action_std=jnp.asarray(action_std, dtype=jnp.float32),
            action_low=jnp.asarray(action_low, dtype=jnp.float32),
            action_high=jnp.asarray(action_high, dtype=jnp.float32),
            batch_stats=batch_stats,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=tuple(image_keys),
                bc_image_keys=tuple(bc_image_keys),
                residual_extra_image_keys=tuple(residual_extra_image_keys or ()),
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                residual_action_dim=int(residual_action_dim),
                hand_prior_source=str(hand_prior_source),
                hand_core_dim=int(hand_core_dim),
                core_action_dim=int(core_action_dim),
                residual_scale_arm=float(residual_scale_arm),
                residual_scale_core=float(residual_scale_core),
                zero_delta=bool(zero_delta),
                normalize_hand_history=bool(normalize_hand_history),
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        env_actions: jnp.ndarray,
        *,
        bc_model: nn.Module,
        bc_variables: flax.core.FrozenDict,
        action_mean: jax.Array,
        action_std: jax.Array,
        action_low: jax.Array,
        action_high: jax.Array,
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        bc_image_keys: Iterable[str] = ("global", "wrist"),
        residual_extra_image_keys: Iterable[str] = (),
        residual_extra_encoder_type: str = "bc-aligned",
        freeze_residual_extra_encoders: bool = True,
        bc_backbone_type: str = "hf_resnet18",
        bc_backbone_config: Optional[dict[str, Any]] = None,
        bc_backbone_params=None,
        bc_backbone_batch_stats=None,
        bc_hil_serl_root: str = DEFAULT_HIL_SERL_ROOT,
        bc_hil_serl_pooling_method: str = "spatial_learned_embeddings",
        bc_hil_serl_num_spatial_blocks: int = 8,
        bc_hil_serl_bottleneck_dim: int = 256,
        augmentation_function: Optional[callable] = None,
        residual_action_dim: Optional[int] = None,
        **kwargs,
    ):
        """Create residual SAC using frozen BCVAE features for actor and critic."""
        residual_extra_image_keys = tuple(residual_extra_image_keys or ())
        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        if residual_action_dim is None:
            residual_action_dim = int(kwargs.get("core_action_dim", 8))

        if residual_extra_encoder_type == "bc-aligned":
            actor_extra_encoders = _make_bc_aligned_extra_visual_encoders(
                residual_extra_image_keys,
                backbone_type=bc_backbone_type,
                backbone_config=bc_backbone_config,
                hil_serl_root=bc_hil_serl_root,
                hil_serl_pooling_method=bc_hil_serl_pooling_method,
                hil_serl_num_spatial_blocks=bc_hil_serl_num_spatial_blocks,
                hil_serl_bottleneck_dim=bc_hil_serl_bottleneck_dim,
            )
            critic_extra_encoders = _make_bc_aligned_extra_visual_encoders(
                residual_extra_image_keys,
                backbone_type=bc_backbone_type,
                backbone_config=bc_backbone_config,
                hil_serl_root=bc_hil_serl_root,
                hil_serl_pooling_method=bc_hil_serl_pooling_method,
                hil_serl_num_spatial_blocks=bc_hil_serl_num_spatial_blocks,
                hil_serl_bottleneck_dim=bc_hil_serl_bottleneck_dim,
            )
            extra_enable_stacking = False
        elif residual_extra_encoder_type == "serl":
            actor_extra_encoders = _make_extra_visual_encoders(
                encoder_type,
                residual_extra_image_keys,
            )
            critic_extra_encoders = _make_extra_visual_encoders(
                encoder_type,
                residual_extra_image_keys,
            )
            extra_enable_stacking = True
        else:
            raise ValueError(
                f"Unknown residual_extra_encoder_type={residual_extra_encoder_type!r}"
            )
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic,
            encoder=BCCoreActionFeatureEncoder(
                residual_extra_image_keys=residual_extra_image_keys,
                encoder=critic_extra_encoders,
                enable_stacking=extra_enable_stacking,
                freeze_extra_encoders=freeze_residual_extra_encoders,
            ),
            network=critic_backbone,
        )(name="critic")

        bc_feature_observations = _make_residual_feature_inputs(
            bc_model=bc_model,
            bc_variables=bc_variables,
            observations=observations,
            action_mean=action_mean,
            action_std=action_std,
            bc_image_keys=bc_image_keys,
            residual_extra_image_keys=residual_extra_image_keys,
            normalize_hand_history=bool(kwargs.get("normalize_hand_history", False)),
            hand_history_std_floor=float(kwargs.get("hand_history_std_floor", 1e-6)),
            zero_delta=bool(kwargs.get("zero_delta", False)),
        )

        policy_def = Policy(
            encoder=BCCoreActionFeatureEncoder(
                residual_extra_image_keys=residual_extra_image_keys,
                encoder=actor_extra_encoders,
                enable_stacking=extra_enable_stacking,
                freeze_extra_encoders=freeze_residual_extra_encoders,
            ),
            network=MLP(**policy_network_kwargs),
            action_dim=residual_action_dim,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            env_actions,
            bc_model=bc_model,
            bc_variables=bc_variables,
            action_mean=action_mean,
            action_std=action_std,
            action_low=action_low,
            action_high=action_high,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            actor_observations=bc_feature_observations,
            critic_observations=bc_feature_observations,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            bc_image_keys=bc_image_keys,
            residual_extra_image_keys=residual_extra_image_keys,
            augmentation_function=augmentation_function,
            residual_action_dim=residual_action_dim,
            residual_extra_encoder_type=residual_extra_encoder_type,
            freeze_residual_extra_encoders=freeze_residual_extra_encoders,
            bc_backbone_type=bc_backbone_type,
            **kwargs,
        )

        if residual_extra_image_keys and residual_extra_encoder_type == "bc-aligned":
            agent = _load_residual_extra_bc_backbone_params(
                agent,
                residual_extra_image_keys,
                backbone_type=bc_backbone_type,
                bc_backbone_params=bc_backbone_params,
                bc_backbone_batch_stats=bc_backbone_batch_stats,
            )
        elif residual_extra_image_keys and "pretrained" in encoder_type:
            agent = _load_residual_extra_resnet10_params(
                agent,
                residual_extra_image_keys,
            )

        return agent

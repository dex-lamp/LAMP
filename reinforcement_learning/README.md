# Reinforcement Learning Supplement

This directory contains the public residual-RL algorithm code used for LAMP
Stage 3. It intentionally excludes private robot interaction infrastructure,
including hardware server code, task reset wrappers, reward services, hardware
IPs, and actor/learner launch scripts tied to a specific lab setup.

## Contents

- `residual_rl/agents/continuous/residual_sac.py`: residual SAC/RLPD agent.
- `residual_rl/agents/continuous/sac.py`: base SAC implementation used by the
  residual agent.
- `residual_rl/networks/`: actor, critic, MLP, temperature, BC policy, and VAE
  model definitions.
- `residual_rl/utils/bc_vae.py`: BC checkpoint loading and observation-to-model
  preprocessing.
- `residual_rl/utils/factory.py`: public factory for constructing a residual
  SAC agent from a frozen BC checkpoint.
- `residual_rl/wrappers/bc_vae_context.py`: environment-agnostic context
  tracking for previous arm action and hand history.

## Algorithm Boundary

The residual actor refines the frozen behavior-cloning policy in the same
action interface used during imitation learning. Arm residuals are applied in
the native arm space, while hand residuals are applied in the compact latent
space before the frozen decoder produces the executable hand command:

```text
bc_core_action = frozen_bc(obs)["core_action"]
residual       = residual_actor(obs, bc_core_action)
adapted_core   = bc_core_action + residual_scale * residual
env_action     = frozen_bc.decode_core_action(adapted_core)
```

This keeps online exploration close to demonstrated, contact-consistent hand
motion instead of perturbing each hand joint independently.

The environment action is assumed to be:

```text
[arm_action_6, hand_delta_6]
```

The public code assumes the caller supplies an environment or replay dataset
with image/state observations and the BC context keys:

```text
bc_policy_state, past_hand_win, current_hand_abs
```

The helper `BCVAEPolicyContext` can generate those keys online or offline
without depending on robot transport.

## Not Included

This supplement copy does not include:

- HTTP or realtime robot server clients.
- Task-specific reset, compliance, or safety wrappers.
- Reward classifier checkpoints or services.
- Lab-specific actor/learner launch scripts.
- Private demo data, checkpoints, or output logs.

## Minimal Usage Sketch

Core dependencies are the same JAX RL stack used by the rest of this project:
`jax`, `flax`, `optax`, `chex`, `distrax`, `gymnasium`, `einops`, and `numpy`.
`transformers` is needed when loading HuggingFace ResNet-18 BC checkpoints.

```python
import jax
import numpy as np

from reinforcement_learning.residual_rl.utils.bc_vae import load_bc_vae_checkpoint
from reinforcement_learning.residual_rl.utils.factory import make_residual_sac_pixel_agent

bc_payload = load_bc_vae_checkpoint("pretrained_models/jax_ckpt/behavior_clone")

agent = make_residual_sac_pixel_agent(
    jax.random.PRNGKey(0),
    sample_obs,
    sample_action,
    action_low=np.full((12,), -1.0, dtype=np.float32),
    action_high=np.full((12,), 1.0, dtype=np.float32),
    bc_payload=bc_payload,
    image_keys=("global", "wrist"),
    bc_image_keys=bc_payload.bc_image_keys,
    residual_scale_arm=0.05,
    residual_scale_core=0.05,
)
```

Run the import smoke check from the repository root:

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

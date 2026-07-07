# Residual RL

The public residual-RL code is designed as an algorithm package, not a complete
robot deployment stack. It assumes the caller owns the environment, replay
buffers, safety checks, reward logic, and actor/learner launch scripts.

## Action Interface

The residual actor refines the frozen BC policy in core-action space:

```text
bc_core_action = frozen_bc(obs)["core_action"]
residual       = residual_actor(obs, bc_core_action)
adapted_core   = bc_core_action + residual_scale * residual
env_action     = frozen_bc.decode_core_action(adapted_core)
```

For the LAMP path, the core action is 8D: six native arm dimensions and two
latent hand dimensions. The final hand command is decoded through the frozen
latent motion prior before execution.

## Required Observation Context

The RL code expects observations to include the image/state keys required by
the frozen BC policy plus hand-history context:

```text
bc_policy_state
past_hand_win
current_hand_abs
```

`reinforcement_learning/residual_rl/wrappers/bc_vae_context.py` provides an
environment-agnostic helper for tracking this context online or offline.

## Minimal Construction Sketch

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

## Attribution

This package follows RLPD-style online/offline replay mixing and SERL/HiL-SERL
JAX robotics RL conventions while adding the LAMP residual action interface.
Please cite the LAMP release and the relevant upstream projects when using this
code:

- RLPD: https://github.com/ikostrikov/rlpd
- SERL: https://github.com/rail-berkeley/serl
- HiL-SERL: https://hil-serl.github.io/

## Smoke Check

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

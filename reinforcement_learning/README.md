# Reinforcement Learning

`reinforcement_learning/` contains the public Stage 3 residual-RL algorithm
code. It adapts a frozen BC policy by adding residuals in the BC core-action
space and then decoding the final action through the same hand interface used
during imitation learning.

```text
bc_core_action = frozen_bc(obs)["core_action"]
residual       = residual_actor(obs, bc_core_action)
adapted_core   = bc_core_action + residual_scale * residual
env_action     = frozen_bc.decode_core_action(adapted_core)
```

## Included

| Path | Purpose |
| --- | --- |
| `residual_rl/agents/continuous/residual_sac.py` | Residual SAC/RLPD agent. |
| `residual_rl/agents/continuous/sac.py` | Base SAC implementation. |
| `residual_rl/networks/` | Actor, critic, BC policy, VAE, and visual model definitions. |
| `residual_rl/utils/` | BC checkpoint loading and residual agent factory helpers. |
| `residual_rl/wrappers/bc_vae_context.py` | Environment-agnostic tracking of hand-history context. |

## Not Included

This package is not a standalone robot launcher. It excludes robot server
clients, task reset wrappers, reward services, actor/learner launch scripts,
private demos, checkpoints, and safety infrastructure.

The implementation follows RLPD-style offline/online replay mixing and
SERL/HiL-SERL-style JAX robotics RL conventions. Cite those projects when
using or discussing this code path:

- RLPD: https://github.com/ikostrikov/rlpd
- SERL: https://github.com/rail-berkeley/serl
- HiL-SERL: https://hil-serl.github.io/

Run the smoke check from the repository root:

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

For integration details, see [`../docs/residual_rl.md`](../docs/residual_rl.md).

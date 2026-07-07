# Common Imitation-Learning Components

`common/` holds shared modules used by the public BC implementation.

## Main Files

| File | Purpose |
| --- | --- |
| `model/bc_dataset.py` | Loads `trajectory_*_demo_expert.pt` files and builds one-step next-action samples. |
| `model/policy_jax.py` | JAX/Flax BC policy with LAMP, raw, PCA, decoder-only, and VQ-VAE hand heads. |
| `model/checkpoint_compat.py` | Checkpoint loading, saving, and compatibility helpers. |

`BCDataset` treats `actions[t]` as the current absolute robot pose and trains
the policy to predict `actions[t + 1]`. For `t == T - 1`, the target holds the
last pose.

The policy can load either HuggingFace ResNet-18 weights or a SERL-style
ResNet-10 backbone when the corresponding checkpoint and code path are
available.

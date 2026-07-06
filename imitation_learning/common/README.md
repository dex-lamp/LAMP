# Common Imitation-Learning Components

This directory contains shared JAX/Flax components used by the LAMP
behavior-cloning path.

## Dataset

`model/bc_dataset.py` provides:

- `compute_action_stats(data_dir)`: action mean/std computation over trajectory
  files.
- `BCDataset`: a single-step next-pose dataset.

Each dataset item includes image tensors, normalized policy state, hand history
windows, and the next-step ground-truth action.

## Policy Modules

`model/policy_jax.py` provides:

- dense layers with source-checkpoint-compatible initialization;
- state encoders (`mlp`, `linear64`, `raw`);
- HuggingFace ResNet-18 and SERL-style ResNet-10 backbones;
- `CoreActionHead`;
- `BCPolicy` with `vae`, `mlp_direct`, `decoder_only`, `vq_codebook`, and
  `pca_raw` hand-prior modes.

## Checkpoints

`model/checkpoint_compat.py` provides JAX pickle checkpoint helpers and
conversion utilities needed by the policy code.

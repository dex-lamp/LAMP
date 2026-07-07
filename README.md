# LAMP Research Code

This repository contains the public research code for "LAMP: Latent Motion
Prior-Guided Real-World Learning for Dexterous Hand Manipulation." It provides
the algorithmic components for learning a compact hand-action interface,
training an imitation policy on top of that interface, and refining the policy
with residual reinforcement learning.

The release is intentionally focused on reusable learning code. Demonstration
data, trained checkpoints, robot server code, calibration files, experiment
logs, reward services, and lab-specific launch scripts are not included.

## Publication Status

The paper is not yet available through arXiv or another public archival source.
Until a public paper record exists, please cite the repository or commit hash
directly when using this code. The paper citation should be added here once the
manuscript is public.

## Method Components

LAMP exposes high-dimensional dexterous hand motion through a compact,
history-conditioned latent action space. The code mirrors the three-stage
pipeline:

- Stage 1, latent motion prior: `vae/` trains a hand-action VAE that maps recent
  hand-action history into a compact latent prior and decodes latent vectors
  back to executable hand targets.
- Stage 2, imitation learning: `imitation_learning/behavior_clone/` trains a
  visuomotor policy that predicts arm commands in the native arm space and hand
  corrections as latent offsets around the learned prior.
- Stage 3, residual RL: `reinforcement_learning/` contains residual SAC/RLPD
  components that add online residuals in the same shared latent hand-action
  interface before decoding the final hand command.

The repository also includes PCA and VQ-VAE baselines for raw, linear, and
discrete hand-action interfaces.

## Repository Layout

- `vae/`: JAX/Flax latent motion prior for hand actions.
- `vq-vae/`: JAX/Flax residual VQ-VAE hand-action tokenizer baseline.
- `pca/`: PCA utilities for low-dimensional hand-action baselines.
- `imitation_learning/`: behavior-cloning code for LAMP Stage 2 and shared
  imitation-learning modules.
- `reinforcement_learning/`: residual SAC/RLPD algorithm code separated from
  robot interaction infrastructure.
- `scripts/`: data conversion utilities for the public trajectory format.

Run commands from the repository root unless a subdirectory README says
otherwise. The `vq-vae/` directory keeps its historical hyphenated name for
checkpoint and script compatibility; use it through the documented scripts
rather than importing it as a Python package.

## Data Convention

Training scripts expect trajectory files named `trajectory_*_demo_expert.pt`.
Each trajectory should provide:

- `actions[:, 0, :]`: a `(T, 12)` action array. The first six dimensions are
  arm commands and the last six dimensions are hand commands.
- `curr_obs["main_images"][:, 0]`: primary RGB camera images.
- `curr_obs["extra_view_images"][:, 0, 0]`: secondary RGB camera images.

Place data under a local directory such as
`data/example_task/demos/success/{train,test}` or pass explicit relative paths
with command line flags. The `data/`, `outputs/`, `visualizations/`, and
`pretrained_models/` directories are ignored by Git.

## Setup

Use Python 3.10 or newer. Install the platform-appropriate JAX build for your
CPU/GPU environment, then install this repository in editable mode:

```bash
pip install -e .
```

For development checks:

```bash
pip install -e ".[dev]"
make check
```

The behavior-cloning examples use the public HuggingFace
`microsoft/resnet-18` checkpoint by default. In offline environments, pass a
local directory containing the same HuggingFace checkpoint files through
`--resnet_path` or `RESNET_PATH`.

## Examples

Train the latent motion prior:

```bash
python vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vae_example
```

Train the Stage 2 behavior-cloning policy:

```bash
TRAIN_DIR=data/example_task/demos/success/train \
TEST_DIR=data/example_task/demos/success/test \
VAE_CKPT=pretrained_models/jax_ckpt/hand_vae \
RESNET_PATH=microsoft/resnet-18 \
OUTPUT_DIR=outputs/behavior_clone_example \
bash imitation_learning/behavior_clone/scripts/train_example_jax.sh
```

Check the public residual-RL imports:

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

## Release Boundary

This repository should not contain experiment logs, absolute local filesystem
paths, robot IP addresses, private user names, private emails, local manuscript
drafts, private datasets, or lab-specific robot interaction code. Robot
deployment requires a separate environment adapter that supplies observations,
rewards, resets, and safety handling.

## License

This code is released under the MIT License. See `LICENSE` for details.

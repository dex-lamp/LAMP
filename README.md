# LAMP Supplement Code

This anonymized repository accompanies the paper "LAMP: Latent Motion
Prior-Guided Real-World Learning for Dexterous Hand Manipulation." It contains
the public algorithmic components for learning a compact hand-action interface,
training an imitation policy on top of that interface, and refining the policy
with residual reinforcement learning.

The release is organized around reusable learning components. Demonstration
data, checkpoints, robot server code, calibration files, experiment logs, reward
services, and lab-specific launch scripts are intentionally not included.

## Paper Alignment

LAMP exposes high-dimensional dexterous hand motion through a compact,
history-conditioned latent action space. The code mirrors the three-stage
pipeline described in the paper:

- Stage 1, latent motion prior: `vae/` trains a hand-action VAE that maps recent
  hand-action history into a compact latent prior and decodes latent vectors
  back to executable hand targets.
- Stage 2, imitation learning: `imitation_learning/behavior_clone/` trains a
  visuomotor policy that predicts arm commands in the native arm space and hand
  corrections as latent offsets around the learned prior.
- Stage 3, residual RL: `reinforcement_learning/` contains residual SAC/RLPD
  components that add online residuals in the same shared latent hand-action
  interface before decoding the final hand command.

The paper also compares this latent interface against raw, linear, and discrete
hand-action interfaces. The public code therefore keeps small baseline modules
for PCA and VQ-VAE style hand-action representations.

## Layout

- `vae/`: JAX/Flax latent motion prior for hand actions.
- `vq-vae/`: JAX/Flax residual VQ-VAE hand-action tokenizer baseline.
- `pca/`: PCA utilities for low-dimensional hand-action baselines.
- `imitation_learning/`: behavior-cloning code for LAMP Stage 2 and a
  hand-only BC baseline.
- `reinforcement_learning/`: residual SAC/RLPD algorithm code separated from
  robot interaction infrastructure.
- `scripts/`: data conversion utilities for the public trajectory format.

## Data Convention

Training scripts expect trajectory files named `trajectory_*_demo_expert.pt`.
Each trajectory should provide:

- `actions[:, 0, :]`: a `(T, 12)` action array. The first six dimensions are
  arm commands and the last six dimensions are hand commands.
- `curr_obs["main_images"][:, 0]`: primary RGB camera images.
- `curr_obs["extra_view_images"][:, 0, 0]`: secondary RGB camera images.

The scripts use relative paths by default. Place data under a local directory
such as `data/example_task/demos/success/{train,test}` or pass explicit
relative paths with command line flags.

## Setup

Use a JAX-capable Python environment with `jax`, `flax`, `optax`, `chex`,
`distrax`, `gymnasium`, `einops`, `numpy`, `torch`, and `transformers`.

```bash
pip install -e .
```

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
OUTPUT_DIR=outputs/behavior_clone_example \
bash imitation_learning/behavior_clone/scripts/train_example_jax.sh
```

Check the public residual-RL imports:

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

## Privacy Boundary

This branch should not contain experiment logs, absolute local filesystem
paths, robot IP addresses, private user names, private emails, local manuscript
drafts, or lab-specific robot interaction code. Robot deployment requires a
separate environment adapter that supplies observations, rewards, resets, and
safety handling.

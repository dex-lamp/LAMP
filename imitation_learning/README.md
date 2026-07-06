# Imitation Learning

This directory contains the public imitation-learning code used for LAMP Stage
2. The policy learns from demonstrations after the latent motion prior has been
pretrained. It predicts arm commands directly and predicts hand commands through
a compact latent interface supplied by the frozen hand-action prior.

## Contents

- `behavior_clone/`: integrated visuomotor behavior cloning over arm commands
  and latent-prior-guided hand actions.
- `common/`: shared datasets, policy modules, and checkpoint helpers used by
  the behavior-cloning implementation.

## Shared Data Format

All policies read trajectory files named `trajectory_*_demo_expert.pt`.
The expected fields are:

- `actions[:, 0, :]`: `(T, 12)` actions.
- `curr_obs["main_images"][:, 0]`: primary RGB images.
- `curr_obs["extra_view_images"][:, 0, 0]`: secondary RGB images.

Behavior cloning follows a next-pose convention: a sample at time `t` uses the
current observation and action history to predict the action at `t + 1`.

## Backbones

The behavior-cloning policy supports:

- `hf_resnet18`: HuggingFace Flax ResNet-18 loaded from `--resnet_path`.
- `hil_serl_resnet10`: a SERL-style ResNet-10 loaded from `HIL_SERL_ROOT` or
  `--hil_serl_root`, with weights passed through `--resnet10_ckpt`.

For anonymous release builds, prefer explicit relative checkpoint paths.

## Generic Behavior-Cloning Launch

```bash
TRAIN_DIR=data/example_task/demos/success/train \
TEST_DIR=data/example_task/demos/success/test \
VAE_CKPT=pretrained_models/jax_ckpt/hand_vae \
RESNET_PATH=pretrained_models/resnet-18 \
OUTPUT_DIR=outputs/behavior_clone_example \
bash imitation_learning/behavior_clone/scripts/train_example_jax.sh
```

The same configuration can be passed directly to
`imitation_learning/behavior_clone/scripts/train_jax.py`.

## Evaluation

```bash
python imitation_learning/behavior_clone/scripts/eval_jax.py \
  --ckpt_dir outputs/behavior_clone_example \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/behavior_clone_example
```

Online control requires a separate adapter that maps environment observations
into the dataset format and handles deployment-specific transport, safety, and
reset logic.

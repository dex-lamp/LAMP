# Behavior Cloning

`behavior_clone/` implements the LAMP Stage 2 policy. It trains a single-step
visuomotor policy that predicts a 6D arm command and a hand command expressed
through the latent motion prior learned in Stage 1.

## Model

The public implementation reuses shared modules from
`imitation_learning/common/model/`.

Supported visual backbones:

- `hf_resnet18`
- `hil_serl_resnet10`

Supported hand heads:

- `vae`: encode hand history with a frozen VAE prior, predict `delta_z`, and
  decode `mu_prior + delta_z`.
- `mlp_direct`: predict the full 12D action directly.
- `decoder_only`: predict `[arm_action_6d, z_2d]` and decode hand action with a
  frozen VAE decoder.
- `vq_codebook`: predict a normalized code index and look up a hand codebook.
- `pca_raw`: predict low-dimensional PCA coordinates and invert them to a hand
  action.

The `vae` mode is the LAMP path: it keeps imitation learning close to a
demonstration-supported hand-motion manifold while allowing the visual policy
to adjust the local latent offset for the current scene.

## Data

Training data should be stored in relative directories such as:

```text
data/example_task/demos/success/train
data/example_task/demos/success/test
```

Each directory should contain `trajectory_*_demo_expert.pt` files.

## Train

Use the generic launch script:

```bash
TRAIN_DIR=data/example_task/demos/success/train \
TEST_DIR=data/example_task/demos/success/test \
VAE_CKPT=pretrained_models/jax_ckpt/hand_vae \
RESNET_PATH=microsoft/resnet-18 \
OUTPUT_DIR=outputs/behavior_clone_example \
bash imitation_learning/behavior_clone/scripts/train_example_jax.sh
```

Or call Python directly:

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --vae_ckpt pretrained_models/jax_ckpt/hand_vae \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_example \
  --backbone_impl hf_resnet18 \
  --hand_prior_source vae \
  --batch_size 128 \
  --total_steps 20000
```

## Evaluate

```bash
python imitation_learning/behavior_clone/scripts/eval_jax.py \
  --ckpt_dir outputs/behavior_clone_example \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/behavior_clone_example
```

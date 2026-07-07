# Behavior Cloning

`behavior_clone/` is the Stage 2 training entry point. It trains a single-step
visuomotor policy over two or more RGB views, the current 12D robot pose, and
recent hand-action history.

## Hand Interfaces

| `--hand_prior_source` | Meaning |
| --- | --- |
| `vae` | Full LAMP mode. Predict a 6D arm command plus a latent hand offset, then decode through the frozen LMPM. |
| `mlp_direct` | Directly regress the full 12D next action. |
| `decoder_only` | Predict a 2D latent hand command and decode through the frozen LMPM without the prior offset. |
| `vq_codebook` | Predict a normalized VQ code index and look up a 6D hand action. Cite DQ-RISE when reporting this baseline. |
| `pca_raw` | Predict 2D PCA hand coordinates and invert them to a 6D hand action. |

## Train

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --vae_ckpt pretrained_models/jax_ckpt/hand_vae \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_example \
  --backbone_impl hf_resnet18 \
  --hand_prior_source vae
```

The shell wrapper exposes the same default path:

```bash
TRAIN_DIR=data/example_task/demos/success/train \
TEST_DIR=data/example_task/demos/success/test \
VAE_CKPT=pretrained_models/jax_ckpt/hand_vae \
RESNET_PATH=microsoft/resnet-18 \
OUTPUT_DIR=outputs/behavior_clone_example \
bash imitation_learning/behavior_clone/scripts/train_example_jax.sh
```

## Evaluate

```bash
python imitation_learning/behavior_clone/scripts/eval_jax.py \
  --ckpt_dir outputs/behavior_clone_example \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/behavior_clone_example
```

Online deployment requires an external adapter for observations, action
transport, resets, reward handling, and safety checks.

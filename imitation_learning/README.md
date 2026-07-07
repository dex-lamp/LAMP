# Imitation Learning

`imitation_learning/` contains the Stage 2 behavior-cloning code. The policy
uses RGB observations, robot state, and recent hand-action history to predict
the next robot action.

## Layout

| Path | Role |
| --- | --- |
| `behavior_clone/` | Training and evaluation entry points for the integrated visuomotor policy. |
| `common/` | Shared datasets, policy modules, checkpoint helpers, and visual backbones. |

The LAMP path uses `--hand_prior_source vae`: the policy predicts a native 6D
arm command and a 2D latent hand offset, then decodes the hand command through
the frozen latent motion prior.

Other supported hand interfaces are documented in
[`../docs/baselines.md`](../docs/baselines.md).

## Train

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

For data conventions and the full training sequence, see
[`../docs/data_format.md`](../docs/data_format.md) and
[`../docs/training.md`](../docs/training.md).

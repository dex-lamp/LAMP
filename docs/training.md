# Training

This page summarizes the public training flow. Run commands from the repository
root.

## Stage 0: Prepare Data

Place demonstrations under:

```text
data/example_task/demos/success/train
data/example_task/demos/success/test
```

See [`data_format.md`](data_format.md) for required keys and tensor shapes.

## Stage 1: Train the Latent Motion Prior

```bash
python vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vae_example
```

The checkpoint is written to:

```text
outputs/hand_vae_example/jax_checkpoints/checkpoint.pkl
```

For downstream BC, either pass that directory directly or copy/link it under a
local checkpoint path such as:

```text
pretrained_models/jax_ckpt/hand_vae
```

## Stage 2: Train Behavior Cloning

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --vae_ckpt outputs/hand_vae_example/jax_checkpoints \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_example \
  --backbone_impl hf_resnet18 \
  --hand_prior_source vae
```

Evaluate the trained policy:

```bash
python imitation_learning/behavior_clone/scripts/eval_jax.py \
  --ckpt_dir outputs/behavior_clone_example \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/behavior_clone_example
```

## Baselines

Action-interface baselines use the same BC trainer with a different
`--hand_prior_source`:

```text
mlp_direct    direct 12D action regression
pca_raw       2D PCA hand-action subspace
vq_codebook   DQ-RISE-style discrete hand-action codebook
decoder_only  VAE decoder without the LAMP prior-offset interface
```

See [`baselines.md`](baselines.md) for commands and attribution notes.

## Stage 3: Residual RL

The public repository includes residual SAC/RLPD model and agent code, but not
the lab-specific robot environment or actor/learner launch scripts. The RL
package expects the caller to provide replay data or an environment with image
observations, state observations, reward handling, resets, and safety logic.

Validate imports with:

```bash
python reinforcement_learning/residual_rl/scripts/smoke_test_imports.py
```

See [`residual_rl.md`](residual_rl.md) for the integration boundary.

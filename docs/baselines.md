# Baselines

The BC trainer supports multiple hand-action interfaces through
`--hand_prior_source`. Keep all other settings fixed when using these modes for
controlled comparisons.

## Raw 12D Regression

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --hand_prior_source mlp_direct \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_raw_example
```

`mlp_direct` predicts the full 12D next action directly.

## PCA Hand Subspace

Fit PCA over raw 6D hand targets:

```bash
python pca/scripts/fit_pca.py \
  --data-dir data/example_task/demos \
  --output-dir outputs/hand_pca_example \
  --save-plot
```

Train BC with the fitted PCA model:

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --hand_prior_source pca_raw \
  --hand_pca_model outputs/hand_pca_example/pca_model.npz \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_pca_example
```

The public policy currently uses a 2D PCA hand coordinate.

## VQ-VAE / DQ-RISE-Style Codebook

Train the hand VQ-VAE:

```bash
python vq-vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vqvae_example
```

Export the codebook:

```bash
python vq-vae/scripts/export_codebook_jax.py \
  --ckpt_dir outputs/hand_vqvae_example \
  --output outputs/hand_vqvae_example/sorted_codebook.npy
```

Train BC with codebook prediction:

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --hand_prior_source vq_codebook \
  --hand_codebook outputs/hand_vqvae_example/sorted_codebook.npy \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_vq_example
```

This baseline follows the DQ-RISE-style quantized hand-state idea. Cite the
LAMP release and DQ-RISE when reporting results:
https://github.com/rise-policy/DQ-RISE.

## Decoder-Only Latent Head

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --vae_ckpt outputs/hand_vae_example/jax_checkpoints \
  --hand_prior_source decoder_only \
  --resnet_path microsoft/resnet-18 \
  --output_dir outputs/behavior_clone_decoder_example
```

`decoder_only` uses the frozen VAE decoder but does not use the
history-conditioned prior offset that defines the full LAMP interface.

# VQ-VAE Baseline

`vq-vae/` contains the discrete hand-action interface used for the VQ-VAE
baseline. It trains a residual VQ hand-action autoencoder and exports a sorted
6D hand-action codebook for behavior cloning.

This implementation is a JAX/Flax reproduction of the DQ-RISE-style quantized
hand-state baseline. It follows the same high-level idea of replacing direct
hand regression with codebook prediction, while fitting into the LAMP BC and
residual-RL pipeline. If you use this baseline, cite both the LAMP release and
DQ-RISE: https://github.com/rise-policy/DQ-RISE.

## Train

```bash
python vq-vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vqvae_example
```

## Export Codebook

```bash
python vq-vae/scripts/export_codebook_jax.py \
  --ckpt_dir outputs/hand_vqvae_example \
  --output outputs/hand_vqvae_example/sorted_codebook.npy
```

## Use With Behavior Cloning

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --hand_prior_source vq_codebook \
  --hand_codebook outputs/hand_vqvae_example/sorted_codebook.npy \
  --output_dir outputs/behavior_clone_vq_example
```

For all action-interface baselines, see
[`../docs/baselines.md`](../docs/baselines.md).

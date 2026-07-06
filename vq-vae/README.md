# Hand Residual VQ-VAE

`vq-vae/` contains a JAX/Flax residual VQ-VAE for 6D hand actions. The model
uses two residual vector-quantization layers and exports a sorted hand-action
codebook that can be consumed by behavior cloning.

## Train

```bash
python vq-vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vqvae_example \
  --batch_size 256 \
  --total_steps 20000
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

The model predicts a normalized hand-code index and performs a hard lookup in
the sorted codebook at inference time.

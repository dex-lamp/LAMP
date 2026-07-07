# Latent Motion Prior

`vae/` trains the Stage 1 latent motion prior module (LMPM). The model encodes
an 8-step hand-action history and decodes a compact latent sample back to the
next 6D absolute hand target.

Default settings match the public LAMP setup: `window_size=8`,
`latent_dim=2`, `hidden_dim=256`, `beta=0.001`, and `total_steps=20000`.

## Train

```bash
python vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vae_example
```

## Evaluate

```bash
python vae/scripts/eval_jax.py \
  --ckpt_dir outputs/hand_vae_example/jax_checkpoints \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/hand_vae_example
```

## Outputs

Training writes a JAX pickle checkpoint under:

```text
outputs/hand_vae_example/jax_checkpoints/checkpoint.pkl
```

That checkpoint is consumed by the LAMP BC path through
`--vae_ckpt pretrained_models/jax_ckpt/hand_vae` or an equivalent local path.

For the full training order, see [`../docs/training.md`](../docs/training.md).

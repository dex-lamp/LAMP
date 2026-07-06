# Hand-Action VAE

`vae/` contains a JAX/Flax hand-action VAE. The model consumes a fixed-length
history of absolute hand actions and predicts the next 6D hand action.

## Train

```bash
python vae/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --output_dir outputs/hand_vae_example \
  --latent_dim 2 \
  --batch_size 256 \
  --total_steps 20000
```

## Evaluate

```bash
python vae/scripts/eval_jax.py \
  --ckpt_dir outputs/hand_vae_example/jax_checkpoints \
  --test_dir data/example_task/demos/success/test \
  --output_dir visualizations/hand_vae_example
```

## Checkpoint Layout

Training writes JAX pickle checkpoints under:

```text
outputs/hand_vae_example/jax_checkpoints/checkpoint.pkl
```

The payload stores model arguments, optimizer state, and training arguments.

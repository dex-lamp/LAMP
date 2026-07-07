# PCA Baseline

`pca/` fits a linear hand-action subspace for the PCA baseline. The fitter uses
raw absolute hand targets from `data["actions"][:, 0, 6:12]` and saves
projection metadata consumed by behavior cloning.

## Fit

```bash
python pca/scripts/fit_pca.py \
  --data-dir data/example_task/demos \
  --output-dir outputs/hand_pca_example \
  --save-plot
```

The output directory contains:

- `pca_model.npz`: mean, components, variance, and projection matrices;
- `loss_by_dim.csv`: reconstruction loss by latent dimension;
- `summary.json`: data and fit metadata;
- `reconstruction_loss.png`: optional plot when `--save-plot` is set.

Use the fitted model in behavior cloning with:

```bash
python imitation_learning/behavior_clone/scripts/train_jax.py \
  --train_dir data/example_task/demos/success/train \
  --test_dir data/example_task/demos/success/test \
  --hand_prior_source pca_raw \
  --hand_pca_model outputs/hand_pca_example/pca_model.npz \
  --output_dir outputs/behavior_clone_pca_example
```

See [`../docs/baselines.md`](../docs/baselines.md) for the comparison modes.

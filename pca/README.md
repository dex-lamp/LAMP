# Hand-Action PCA

`pca/` provides a small PCA utility for low-dimensional hand-action
representations.

## Fit

```bash
python pca/scripts/fit_pca.py \
  --data-dir data/example_task/demos \
  --output-dir outputs/hand_pca_example \
  --n-components 2
```

The output directory contains `pca_model.npz` with:

- `mean`: hand-action mean with shape `(6,)`;
- `components`: PCA components with shape `(n_components, 6)`;
- `explained_variance`;
- `explained_variance_ratio`.

The resulting file can be passed to behavior cloning with
`--hand_prior_source pca_raw --hand_pca_model outputs/hand_pca_example/pca_model.npz`.

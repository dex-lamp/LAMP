# Data Format

All public training scripts expect PyTorch trajectory files named:

```text
trajectory_*_demo_expert.pt
```

A typical task directory is:

```text
data/example_task/demos/
  success/
    train/
      trajectory_0_demo_expert.pt
    test/
      trajectory_1_demo_expert.pt
  failure/
    trajectory_2_demo_expert.pt
```

The core training scripts use the successful train/test split. Failure data is
kept for conversion, inspection, or custom RL workflows.

## Required Keys

Each trajectory file should load to a dictionary with at least:

```python
data["actions"]                         # Tensor, shape (T, 1, 12)
data["curr_obs"]["main_images"]         # Tensor, shape (T, 1, H, W, C)
data["curr_obs"]["extra_view_images"]   # Tensor, shape (T, 1, V, H, W, C)
```

Action dimensions are split as:

```text
actions[:, 0, 0:6]   -> arm command / arm pose
actions[:, 0, 6:12]  -> hand command / absolute hand pose
```

Images are converted to channel-first tensors inside the dataset code. The
default BC image keys are:

```text
global -> curr_obs["main_images"][:, 0]
wrist  -> curr_obs["extra_view_images"][:, 0, 0]
```

`global_2` is also supported when `extra_view_images` contains a second extra
view.

## Prediction Convention

The public BC and VAE code treats `actions[t]` as the current absolute robot
pose and predicts the next absolute pose:

```text
observation at t        -> actions[t]
supervised target       -> actions[t + 1]
target at final frame   -> actions[T - 1]
```

The VAE receives a padded 8-step hand-action history
`[hand[t - 7], ..., hand[t]]` and predicts `hand[t + 1]`.

## Conversion Helpers

Convert flattened pickle demonstrations into the public `.pt` layout:

```bash
python scripts/convert_pkl_to_demos.py \
  --input-dir data/example_task_raw \
  --output-dir data/example_task/demos \
  --overwrite
```

Create a delta-action copy of an absolute-action dataset:

```bash
python scripts/convert_to_delta_action.py \
  --input-dir data/example_task/demos \
  --output-dir data/example_task/demos_delta \
  --copy-other-files
```

The default LAMP training path uses the absolute-action convention.

# Installation

LAMP is a Python research-code release built around JAX/Flax, PyTorch data
loading, and HuggingFace Transformers backbones.

## Environment

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the JAX wheel that matches your platform before running large training
jobs. Then install the repository in editable mode:

```bash
pip install -e .
```

For formatting and lint checks:

```bash
pip install -e ".[dev]"
make check
```

## External Checkpoints

Behavior cloning defaults to the public HuggingFace `microsoft/resnet-18`
checkpoint:

```bash
--resnet_path microsoft/resnet-18
```

For offline machines, pass a local directory containing the same checkpoint
files through `--resnet_path` or the `RESNET_PATH` environment variable used by
`imitation_learning/behavior_clone/scripts/train_example_jax.sh`.

## Local Artifacts

The repository intentionally ignores local training artifacts:

```text
data/
outputs/
visualizations/
pretrained_models/
wandb/
rollouts/
```

Keep private demos, checkpoints, robot configuration, and unreleased manuscript
files outside Git.

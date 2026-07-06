"""Trajectory loading helpers for hand VQ-VAE training.

The repository stores demonstrations as PyTorch `.pt` trajectory files, so this
module intentionally keeps `torch.load` at the data boundary. The VQ-VAE model
and optimization code are JAX-only.
"""

from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch


def list_trajectory_files(data_dir: str) -> List[Path]:
    data_dir = Path(data_dir)
    traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    return traj_files


def load_hand_action_array(data_dir: str) -> np.ndarray:
    chunks = []
    traj_files = list_trajectory_files(data_dir)
    for path in traj_files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        chunks.append(data["actions"][:, 0, 6:12].float().numpy())
    actions = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"Loaded {len(traj_files)} trajectories, {len(actions)} hand-action samples from {data_dir}")
    return actions


def iterate_minibatches(
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.RandomState,
) -> Iterator[np.ndarray]:
    indices = np.arange(actions.shape[0])
    if shuffle:
        rng.shuffle(indices)
    if drop_last:
        n_batches = len(indices) // batch_size
        indices = indices[: n_batches * batch_size]
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < batch_size and drop_last:
            continue
        yield actions[batch_idx]

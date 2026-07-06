"""Shared trajectory loading helpers for Torch and JAX VAE codepaths."""

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch


def list_trajectory_files(data_dir: str) -> List[Path]:
    data_dir = Path(data_dir)
    traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    return traj_files


def load_hand_actions(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["actions"][:, 0, 6:12].float()


def build_window_target_pairs(actions: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    windows = []
    targets = []
    num_steps = actions.shape[0]

    for step in range(num_steps):
        start = step - window_size + 1
        if start < 0:
            pad_len = -start
            window = torch.cat(
                [
                    actions[0:1].expand(pad_len, -1),
                    actions[0 : step + 1],
                ],
                dim=0,
            )
        else:
            window = actions[start : step + 1]

        target = actions[step + 1] if step + 1 < num_steps else actions[step]
        windows.append(window)
        targets.append(target)

    return torch.stack(windows), torch.stack(targets)


def load_window_arrays(data_dir: str, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    traj_files = list_trajectory_files(data_dir)
    windows = []
    targets = []
    num_samples = 0

    for path in traj_files:
        hand_actions = load_hand_actions(path)
        traj_windows, traj_targets = build_window_target_pairs(hand_actions, window_size)
        windows.append(traj_windows.numpy())
        targets.append(traj_targets.numpy())
        num_samples += traj_windows.shape[0]

    window_array = np.concatenate(windows, axis=0).astype(np.float32)
    target_array = np.concatenate(targets, axis=0).astype(np.float32)
    print(
        f"Loaded {len(traj_files)} trajectories, {num_samples} samples "
        f"(window={window_size}) from {data_dir}"
    )
    return window_array, target_array


def iterate_minibatches(
    windows: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    rng: np.random.RandomState,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(windows))
    if shuffle:
        rng.shuffle(indices)

    if drop_last:
        num_batches = len(indices) // batch_size
        indices = indices[: num_batches * batch_size]

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if len(batch_indices) < batch_size and drop_last:
            continue
        yield windows[batch_indices], targets[batch_indices]

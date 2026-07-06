"""
Dataset for hand action VAE training.

Each sample is a (window, target) pair:
  window: (window_size, 6) past hand actions [t-7 : t]
  target: (6,) ground truth hand action at t+1

Zero-pads the beginning of each trajectory if fewer than window_size frames available.
For the last frame of each trajectory, target = action at t (hold).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class HandActionWindowDataset(Dataset):
    """Builds sliding window samples from all trajectories.

    Args:
        noise_std: if > 0, add Gaussian noise to the input window during __getitem__.
                   Target is always clean GT. Only effective during training.
    """

    def __init__(self, data_dir: str, window_size: int = 8, noise_std: float = 0.0):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        self.window_size = window_size
        self.noise_std = noise_std
        self.samples = []  # list of (window, target) tensors

        for f in traj_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            actions = data["actions"][:, 0, 6:12].float()  # (T, 6) hand actions
            T = actions.shape[0]

            for t in range(T):
                # Build window [t - window_size + 1 : t + 1]
                start = t - window_size + 1
                if start < 0:
                    # Pad with first frame (instead of zeros, to avoid false "hand open" signal)
                    pad_len = -start
                    window = torch.cat([
                        actions[0:1].expand(pad_len, -1),
                        actions[0:t + 1],
                    ], dim=0)
                else:
                    window = actions[start:t + 1]

                # Target: action at t+1, or action at t for last frame
                if t + 1 < T:
                    target = actions[t + 1]
                else:
                    target = actions[t]

                self.samples.append((window, target))

        print(f"Loaded {len(traj_files)} trajectories, {len(self.samples)} samples "
              f"(window={window_size}) from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, target = self.samples[idx]
        if self.noise_std > 0:
            window = window + torch.randn_like(window) * self.noise_std
        return window, target  # (window_size, 6), (6,)

"""Data loading helpers for single-frame hand6 PCA.

The demonstration files store robot actions as PyTorch tensors.  We keep
``torch.load`` at this boundary and expose NumPy arrays to the PCA code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAJECTORY_PATTERN = "trajectory_*_demo_expert.pt"
HAND_SLICE_DESCRIPTION = 'data["actions"][:, 0, 6:12]'


def _trajectory_files(directory: Path) -> list[Path]:
    return sorted(directory.glob(TRAJECTORY_PATTERN))


def _source_record(label: str, directory: Path, files: list[Path]) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(directory),
        "num_trajectories": len(files),
    }


def discover_trajectory_files(data_dir: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Find trajectory files from a demos root or a direct trajectory directory.

    Supported layouts:
      * ``<data_dir>/trajectory_*_demo_expert.pt``
      * ``<data_dir>/success/{train,test}/trajectory_*_demo_expert.pt``
      * ``<data_dir>/{train,test}/trajectory_*_demo_expert.pt``

    The last form lets callers pass the ``success`` directory directly.  Failure
    trajectories are intentionally ignored for the demos-root layout.
    """

    root = Path(data_dir).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data_dir is not a directory: {root}")

    direct_files = _trajectory_files(root)
    if direct_files:
        return direct_files, {
            "mode": "direct",
            "data_dir": str(root),
            "sources": [_source_record("direct", root, direct_files)],
            "ignored_dirs": [],
        }

    layouts = [
        (
            "demos_success_train_test",
            [("success/train", root / "success" / "train"), ("success/test", root / "success" / "test")],
            [root / "failure"],
        ),
        (
            "split_train_test",
            [("train", root / "train"), ("test", root / "test")],
            [],
        ),
    ]
    for mode, source_dirs, ignored_dirs in layouts:
        files: list[Path] = []
        sources: list[dict[str, Any]] = []
        for label, directory in source_dirs:
            if not directory.is_dir():
                continue
            split_files = _trajectory_files(directory)
            if split_files:
                files.extend(split_files)
                sources.append(_source_record(label, directory, split_files))
        if files:
            return files, {
                "mode": mode,
                "data_dir": str(root),
                "sources": sources,
                "ignored_dirs": [str(path) for path in ignored_dirs if path.exists()],
            }

    expected = (
        f"{root}/{TRAJECTORY_PATTERN} or "
        f"{root}/success/{{train,test}}/{TRAJECTORY_PATTERN}"
    )
    raise FileNotFoundError(f"No trajectory files found. Expected {expected}")


def load_hand_action_file(path: str | Path) -> np.ndarray:
    """Load one trajectory file and return raw absolute hand pose, shape ``(T, 6)``."""

    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "actions" not in data:
        raise KeyError(f"{path} does not contain an 'actions' tensor")

    actions = data["actions"]
    if actions.ndim < 3 or actions.shape[1] < 1 or actions.shape[-1] < 12:
        raise ValueError(
            f"{path} has actions shape {tuple(actions.shape)}, expected at least (T, 1, 12)"
        )

    hand = actions[:, 0, 6:12]
    if isinstance(hand, torch.Tensor):
        hand = hand.detach().cpu().float().numpy()
    else:
        hand = np.asarray(hand, dtype=np.float32)

    hand = np.asarray(hand, dtype=np.float32)
    if hand.ndim != 2 or hand.shape[1] != 6:
        raise ValueError(f"{path} produced hand array shape {hand.shape}, expected (T, 6)")
    if not np.all(np.isfinite(hand)):
        raise ValueError(f"{path} contains non-finite hand action values")
    return hand


def load_hand_action_dataset(data_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load all discovered trajectories and return ``(actions, metadata)``."""

    files, metadata = discover_trajectory_files(data_dir)
    chunks: list[np.ndarray] = []
    per_file: list[dict[str, Any]] = []
    for path in files:
        hand = load_hand_action_file(path)
        chunks.append(hand)
        per_file.append({"path": str(path), "num_frames": int(hand.shape[0])})

    actions = np.concatenate(chunks, axis=0).astype(np.float32)
    metadata = dict(metadata)
    metadata["num_trajectories"] = len(files)
    metadata["num_frames"] = int(actions.shape[0])
    metadata["hand_slice"] = HAND_SLICE_DESCRIPTION
    metadata["files"] = per_file
    return actions, metadata


def load_hand_action_array(data_dir: str | Path) -> np.ndarray:
    """Load only the concatenated hand action array."""

    actions, _ = load_hand_action_dataset(data_dir)
    return actions

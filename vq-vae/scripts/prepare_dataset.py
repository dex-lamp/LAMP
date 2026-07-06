"""
Prepare demo dataset for VQ-VAE training.

1. Split trajectories into success / failure subfolders.
2. For success trajectories: strip leading/trailing zero-action frames.
3. Randomly split into train / test sets.

Usage:
    python scripts/prepare_dataset.py --input_dir data/example_task/demos
    python scripts/prepare_dataset.py --input_dir data/example_task/demos --train_ratio 0.9
    python scripts/prepare_dataset.py --input_dir data/example_task/demos --seed 42
    python scripts/prepare_dataset.py --input_dir data/example_task/demos --strip_mode head
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch


def is_success(data: dict) -> bool:
    """Check if trajectory is successful (reward > 0)."""
    return data["rewards"].sum().item() > 0


def strip_zero_frames(data: dict, mode: str = "head") -> dict:
    """
    Remove zero-action frames from trajectory.

    Args:
        data: trajectory dict loaded from .pt file.
        mode: "head"  - only strip leading consecutive zeros.
              "both"  - strip leading and trailing consecutive zeros.
              "all"   - strip ALL exact-zero frames (including mid-trajectory pauses).

    Returns:
        New trajectory dict with zero frames removed.
    """
    actions = data["actions"][:, 0, :]  # (T, 12)
    T = actions.shape[0]
    norms = actions.norm(dim=1)  # (T,)

    if mode == "head":
        # Find first non-zero frame
        start = 0
        while start < T and norms[start] == 0:
            start += 1
        keep_mask = torch.zeros(T, dtype=torch.bool)
        keep_mask[start:] = True

    elif mode == "both":
        # Find first and last non-zero frame
        start = 0
        while start < T and norms[start] == 0:
            start += 1
        end = T - 1
        while end >= 0 and norms[end] == 0:
            end -= 1
        keep_mask = torch.zeros(T, dtype=torch.bool)
        if start <= end:
            keep_mask[start:end + 1] = True

    elif mode == "all":
        keep_mask = norms > 0

    else:
        raise ValueError(f"Unknown strip_mode: {mode}")

    # If all frames are zero, keep the whole trajectory (edge case)
    if keep_mask.sum() == 0:
        return data

    return _apply_mask(data, keep_mask)


def _apply_mask(data: dict, mask: torch.Tensor) -> dict:
    """Apply boolean mask to all time-indexed tensors in trajectory."""
    new_data = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == mask.shape[0]:
            new_data[key] = val[mask]
        elif isinstance(val, dict):
            new_data[key] = _apply_mask_dict(val, mask)
        else:
            new_data[key] = val
    return new_data


def _apply_mask_dict(d: dict, mask: torch.Tensor) -> dict:
    """Recursively apply mask to nested dict of tensors."""
    new_d = {}
    for key, val in d.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == mask.shape[0]:
            new_d[key] = val[mask]
        elif isinstance(val, dict):
            new_d[key] = _apply_mask_dict(val, mask)
        else:
            new_d[key] = val
    return new_d


def find_trajectories(input_dir: Path) -> list:
    """Find all trajectory .pt files in directory."""
    files = sorted(input_dir.glob("trajectory_*_demo_expert.pt"))
    return files


def save_trajectory(data: dict, output_path: Path):
    """Save trajectory dict to .pt file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare demo dataset for VQ-VAE training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to demos directory containing trajectory .pt files")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of success trajectories for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split (default: 42)")
    parser.add_argument("--strip_mode", type=str, default="head",
                        choices=["head", "both", "all"],
                        help="Zero-frame stripping mode (default: head)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    success_dir = input_dir / "success"
    failure_dir = input_dir / "failure"
    train_dir = success_dir / "train"
    test_dir = success_dir / "test"

    # Find all trajectories
    traj_files = find_trajectories(input_dir)
    print(f"Found {len(traj_files)} trajectories in {input_dir}")

    # Split into success / failure
    success_files = []
    failure_files = []
    for f in traj_files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        if is_success(data):
            success_files.append(f)
        else:
            failure_files.append(f)

    print(f"  Success: {len(success_files)}")
    print(f"  Failure: {len(failure_files)}")

    # --- Save failure trajectories (as-is) ---
    failure_dir.mkdir(parents=True, exist_ok=True)
    for f in failure_files:
        shutil.copy2(f, failure_dir / f.name)
    print(f"\nSaved {len(failure_files)} failure trajectories to {failure_dir}")

    # --- Process success trajectories ---
    # Clean zero frames
    cleaned = []
    total_before = 0
    total_after = 0
    for f in success_files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        T_before = data["actions"].shape[0]
        data_clean = strip_zero_frames(data, mode=args.strip_mode)
        T_after = data_clean["actions"].shape[0]
        total_before += T_before
        total_after += T_after
        cleaned.append((f.name, data_clean))

    removed = total_before - total_after
    print(f"\nZero-frame cleaning (mode={args.strip_mode}):")
    print(f"  Before: {total_before} frames")
    print(f"  After:  {total_after} frames")
    print(f"  Removed: {removed} frames ({removed / total_before * 100:.1f}%)")

    # Random train/test split
    random.seed(args.seed)
    indices = list(range(len(cleaned)))
    random.shuffle(indices)
    n_train = int(len(cleaned) * args.train_ratio)
    train_indices = set(indices[:n_train])

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_frames = 0
    test_frames = 0
    train_files_saved = []
    test_files_saved = []

    for i, (orig_name, data) in enumerate(cleaned):
        if i in train_indices:
            save_trajectory(data, train_dir / orig_name)
            train_frames += data["actions"].shape[0]
            train_files_saved.append(orig_name)
        else:
            save_trajectory(data, test_dir / orig_name)
            test_frames += data["actions"].shape[0]
            test_files_saved.append(orig_name)

    # Save split metadata
    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "strip_mode": args.strip_mode,
        "total_success": len(success_files),
        "total_failure": len(failure_files),
        "n_train": len(train_files_saved),
        "n_test": len(test_files_saved),
        "train_frames": train_frames,
        "test_frames": test_frames,
        "frames_before_cleaning": total_before,
        "frames_after_cleaning": total_after,
        "train_files": sorted(train_files_saved),
        "test_files": sorted(test_files_saved),
    }
    with open(success_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Summary
    print(f"\nTrain/test split (ratio={args.train_ratio}, seed={args.seed}):")
    print(f"  Train: {len(train_files_saved)} trajectories, {train_frames} frames -> {train_dir}")
    print(f"  Test:  {len(test_files_saved)} trajectories, {test_frames} frames -> {test_dir}")
    print(f"\nSplit metadata saved to {success_dir / 'split_info.json'}")
    print("Done!")


if __name__ == "__main__":
    main()

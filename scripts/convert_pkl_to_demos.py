"""Convert flattened pickle demos into trajectory .pt demo folders.

The input pickle files are expected to contain a flat list of transition
dicts with this shape:

    {
        "observations": {"global": ..., "state": ..., "wrist": ...},
        "actions": np.ndarray shape (12,),
        "next_observations": {"global": ..., "state": ..., "wrist": ...},
        "rewards": 0 or 1,
        "dones": 0 or 1,
        "infos": {"succeed": bool, ...},
    }

The output mirrors the existing demo format:

    demos/
      metadata.json
      success/
        split_info.json
        train/trajectory_0_demo_expert.pt
        test/trajectory_1_demo_expert.pt
      failure/
        trajectory_2_demo_expert.pt

By default, the output hand action dimensions (6:12) are filled from the
absolute gripper pose in observations["state"][..., :6]. The raw pickle
action is still used to decide which frames are blank, so static frames can
be removed even though their absolute hand pose is non-zero. By default, blank
frame detection uses the full 12-D raw action, but --clean-action-scope can
limit the check to the wrist (0:6) or hand (6:12) action dimensions.

Example:
    python scripts/convert_pkl_to_demos.py \
        --input-dir data/example_task_raw \
        --output-dir data/example_task_raw/demos \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODEL_WEIGHTS_ID = "demo_expert"


@dataclass
class ConvertedTrajectory:
    index: int
    source_file: str
    source_episode_index: int
    success: bool
    data: dict[str, Any]
    frames_before_cleaning: int
    frames_after_cleaning: int
    blank_frames_removed: int


def load_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a list of transitions, got {type(data)!r}")
    return data


def split_episodes(transitions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for transition in transitions:
        current.append(transition)
        if bool(transition.get("dones", False)):
            episodes.append(current)
            current = []

    if current:
        episodes.append(current)
    return episodes


def transition_success(transition: dict[str, Any]) -> bool:
    if float(transition.get("rewards", 0.0)) > 0.0:
        return True
    infos = transition.get("infos")
    return isinstance(infos, dict) and bool(infos.get("succeed", False))


def episode_success(episode: list[dict[str, Any]]) -> bool:
    return any(transition_success(transition) for transition in episode)


def stack_obs(episode: list[dict[str, Any]], obs_key: str, field: str) -> np.ndarray:
    return np.stack([np.asarray(transition[obs_key][field]) for transition in episode], axis=0)


def stack_raw_actions(episode: list[dict[str, Any]]) -> np.ndarray:
    actions = np.stack(
        [np.asarray(transition["actions"], dtype=np.float32) for transition in episode],
        axis=0,
    )
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected raw actions with shape (T, 12), got {actions.shape}")
    return actions


def stack_gripper_pose(episode: list[dict[str, Any]], obs_key: str) -> np.ndarray:
    poses = []
    for transition in episode:
        infos = transition.get("infos", {})
        original_obs = infos.get("original_state_obs", {}) if isinstance(infos, dict) else {}
        if "gripper_pose" in original_obs:
            pose = original_obs["gripper_pose"]
        else:
            pose = np.asarray(transition[obs_key]["state"])[0, :6]
        poses.append(np.asarray(pose, dtype=np.float32))
    poses_np = np.stack(poses, axis=0)
    if poses_np.ndim != 2 or poses_np.shape[1] != 6:
        raise ValueError(f"Expected gripper poses with shape (T, 6), got {poses_np.shape}")
    return poses_np


def build_output_actions(
    episode: list[dict[str, Any]],
    raw_actions_np: np.ndarray,
    hand_action_source: str,
) -> np.ndarray:
    actions_np = raw_actions_np.copy()
    if hand_action_source == "raw_action":
        return actions_np
    if hand_action_source == "state":
        states = stack_obs(episode, "observations", "state")
        actions_np[:, 6:12] = states[:, 0, :6].astype(np.float32, copy=False)
        return actions_np
    if hand_action_source == "gripper_pose":
        actions_np[:, 6:12] = stack_gripper_pose(episode, "observations")
        return actions_np
    raise ValueError(f"Unknown hand action source: {hand_action_source}")


def to_uint8_tensor(array: np.ndarray) -> torch.Tensor:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array))


def to_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.astype(np.float32, copy=False)))


def format_wrist_images(wrist: torch.Tensor) -> torch.Tensor:
    """Convert stacked wrist images to (T, B, V, H, W, C)."""
    if wrist.ndim == 5:
        return wrist.unsqueeze(2)
    if wrist.ndim == 6:
        return wrist
    raise ValueError(f"Unexpected wrist image shape: {tuple(wrist.shape)}")


def has_observation_field(episode: list[dict[str, Any]], obs_key: str, field: str) -> bool:
    return all(field in transition.get(obs_key, {}) for transition in episode)


def stack_extra_view_images(
    episode: list[dict[str, Any]],
    obs_key: str,
) -> tuple[torch.Tensor, list[str]]:
    """Stack wrist and optional global_2 images as extra_view_images views."""
    view_tensors = [
        format_wrist_images(to_uint8_tensor(stack_obs(episode, obs_key, "wrist")))
    ]
    image_keys = ["wrist"]
    if has_observation_field(episode, obs_key, "global_2"):
        view_tensors.append(
            format_wrist_images(to_uint8_tensor(stack_obs(episode, obs_key, "global_2")))
        )
        image_keys.append("global_2")
    return torch.cat(view_tensors, dim=2), image_keys


def build_trajectory(
    episode: list[dict[str, Any]],
    success: bool,
    max_episode_length: int,
    hand_action_source: str,
) -> dict[str, Any]:
    raw_actions_np = stack_raw_actions(episode)
    actions_np = build_output_actions(
        episode=episode,
        raw_actions_np=raw_actions_np,
        hand_action_source=hand_action_source,
    )

    actions = to_float_tensor(actions_np).unsqueeze(1)
    raw_actions = to_float_tensor(raw_actions_np).unsqueeze(1)
    states = to_float_tensor(stack_obs(episode, "observations", "state"))
    next_states = to_float_tensor(stack_obs(episode, "next_observations", "state"))
    main_images = to_uint8_tensor(stack_obs(episode, "observations", "global"))
    next_main_images = to_uint8_tensor(stack_obs(episode, "next_observations", "global"))
    extra_view_images, extra_image_keys = stack_extra_view_images(episode, "observations")
    next_extra_view_images, next_extra_image_keys = stack_extra_view_images(
        episode,
        "next_observations",
    )
    if extra_image_keys != next_extra_image_keys:
        raise ValueError(
            f"Observation/next_observation image key mismatch: "
            f"{extra_image_keys} vs {next_extra_image_keys}"
        )

    data: dict[str, Any] = {
        "max_episode_length": max_episode_length,
        "model_weights_id": MODEL_WEIGHTS_ID,
        "image_keys": ["global", *extra_image_keys],
        "actions": actions,
        "_raw_actions_for_cleaning": raw_actions,
        "intervene_flags": torch.ones_like(actions, dtype=torch.bool),
        "rewards": torch.zeros((actions.shape[0], 1, 1), dtype=torch.float32),
        "terminations": torch.zeros((actions.shape[0], 1, 1), dtype=torch.bool),
        "truncations": torch.zeros((actions.shape[0], 1, 1), dtype=torch.bool),
        "dones": torch.zeros((actions.shape[0], 1, 1), dtype=torch.bool),
        "forward_inputs": {"action": actions.clone()},
        "curr_obs": {
            "states": states,
            "main_images": main_images,
            "extra_view_images": extra_view_images,
        },
        "next_obs": {
            "states": next_states,
            "main_images": next_main_images,
            "extra_view_images": next_extra_view_images,
        },
    }
    set_terminal_flags(data, success)
    return data


def set_terminal_flags(data: dict[str, Any], success: bool) -> None:
    data["rewards"].zero_()
    data["terminations"].zero_()
    data["truncations"].zero_()
    data["dones"].zero_()

    if success and data["actions"].shape[0] > 0:
        data["rewards"][-1, 0, 0] = 1.0
        data["terminations"][-1, 0, 0] = True
        data["dones"][-1, 0, 0] = True


def select_action_scope(actions: torch.Tensor, clean_action_scope: str) -> torch.Tensor:
    if clean_action_scope == "all":
        return actions
    if actions.shape[1] < 12:
        raise ValueError(
            f"Expected at least 12 action dims for scope {clean_action_scope!r}, "
            f"got {actions.shape[1]}"
        )
    if clean_action_scope == "wrist":
        return actions[:, :6]
    if clean_action_scope == "hand":
        return actions[:, 6:12]
    raise ValueError(f"Unknown clean action scope: {clean_action_scope}")


def blank_action_mask(
    actions: torch.Tensor,
    action_eps: float,
    clean_action_scope: str,
) -> torch.Tensor:
    flat_actions = actions[:, 0, :]
    scoped_actions = select_action_scope(flat_actions, clean_action_scope)
    return torch.linalg.norm(scoped_actions, dim=1) <= action_eps


def apply_time_mask(data: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    new_data: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == mask.shape[:1]:
            new_data[key] = value[mask]
        elif isinstance(value, dict):
            new_data[key] = apply_time_mask(value, mask)
        else:
            new_data[key] = value
    return new_data


def clean_success_blank_frames(
    data: dict[str, Any],
    action_eps: float,
    strip_mode: str,
    clean_action_scope: str,
) -> tuple[dict[str, Any], int]:
    if strip_mode == "none":
        return data, 0

    clean_actions = data.get("_raw_actions_for_cleaning", data["actions"])
    blank = blank_action_mask(clean_actions, action_eps, clean_action_scope)
    if strip_mode == "frame_norm":
        keep = ~blank
    elif strip_mode == "head":
        first_keep = int((~blank).float().argmax().item()) if (~blank).any() else len(blank)
        keep = torch.zeros_like(blank, dtype=torch.bool)
        keep[first_keep:] = True
    elif strip_mode == "both":
        nonblank = torch.where(~blank)[0]
        keep = torch.zeros_like(blank, dtype=torch.bool)
        if len(nonblank) > 0:
            keep[int(nonblank[0]) : int(nonblank[-1]) + 1] = True
    else:
        raise ValueError(f"Unknown strip mode: {strip_mode}")

    if int(keep.sum().item()) == 0:
        return data, 0

    removed = int((~keep).sum().item())
    cleaned = apply_time_mask(data, keep)
    set_terminal_flags(cleaned, success=True)
    cleaned["forward_inputs"]["action"] = cleaned["actions"].clone()
    return cleaned, removed


def drop_internal_fields(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    data.pop("_raw_actions_for_cleaning", None)
    return data


def convert_all(
    input_dir: Path,
    max_episode_length: int,
    action_eps: float,
    strip_mode: str,
    clean_action_scope: str,
    hand_action_source: str,
) -> list[ConvertedTrajectory]:
    pkl_files = sorted(input_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    converted: list[ConvertedTrajectory] = []
    trajectory_index = 0
    for pkl_path in pkl_files:
        episodes = split_episodes(load_pickle(pkl_path))
        for episode_index, episode in enumerate(episodes):
            if not episode:
                continue
            success = episode_success(episode)
            trajectory = build_trajectory(
                episode=episode,
                success=success,
                max_episode_length=max_episode_length,
                hand_action_source=hand_action_source,
            )
            frames_before = int(trajectory["actions"].shape[0])
            removed = 0
            if success:
                trajectory, removed = clean_success_blank_frames(
                    trajectory,
                    action_eps=action_eps,
                    strip_mode=strip_mode,
                    clean_action_scope=clean_action_scope,
                )
            trajectory = drop_internal_fields(trajectory)
            frames_after = int(trajectory["actions"].shape[0])
            converted.append(
                ConvertedTrajectory(
                    index=trajectory_index,
                    source_file=pkl_path.name,
                    source_episode_index=episode_index,
                    success=success,
                    data=trajectory,
                    frames_before_cleaning=frames_before,
                    frames_after_cleaning=frames_after,
                    blank_frames_removed=removed,
                )
            )
            trajectory_index += 1

    return converted


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def trajectory_filename(index: int) -> str:
    return f"trajectory_{index}_demo_expert.pt"


def save_outputs(
    trajectories: list[ConvertedTrajectory],
    output_dir: Path,
    train_ratio: float,
    seed: int,
    input_dir: Path,
    action_eps: float,
    strip_mode: str,
    clean_action_scope: str,
    hand_action_source: str,
) -> dict[str, Any]:
    success = [trajectory for trajectory in trajectories if trajectory.success]
    failure = [trajectory for trajectory in trajectories if not trajectory.success]
    image_keys = (
        list(trajectories[0].data.get("image_keys", ["global", "wrist"]))
        if trajectories
        else ["global", "wrist"]
    )

    success_dir = output_dir / "success"
    failure_dir = output_dir / "failure"
    train_dir = success_dir / "train"
    test_dir = success_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled_indices = list(range(len(success)))
    rng.shuffle(shuffled_indices)
    n_train = int(len(success) * train_ratio)
    train_indices = set(shuffled_indices[:n_train])

    train_files: list[str] = []
    test_files: list[str] = []
    failure_files: list[str] = []
    train_frames = 0
    test_frames = 0
    failure_frames = 0

    manifest: list[dict[str, Any]] = []

    for success_i, trajectory in enumerate(success):
        filename = trajectory_filename(trajectory.index)
        target_dir = train_dir if success_i in train_indices else test_dir
        torch.save(trajectory.data, target_dir / filename)
        if success_i in train_indices:
            train_files.append(filename)
            train_frames += trajectory.frames_after_cleaning
        else:
            test_files.append(filename)
            test_frames += trajectory.frames_after_cleaning
        manifest.append(trajectory_manifest_entry(trajectory, filename, str(target_dir.relative_to(output_dir))))

    for trajectory in failure:
        filename = trajectory_filename(trajectory.index)
        torch.save(trajectory.data, failure_dir / filename)
        failure_files.append(filename)
        failure_frames += trajectory.frames_after_cleaning
        manifest.append(trajectory_manifest_entry(trajectory, filename, "failure"))

    total_frames_after = sum(item.frames_after_cleaning for item in trajectories)
    total_frames_before = sum(item.frames_before_cleaning for item in trajectories)
    blank_frames_removed = sum(item.blank_frames_removed for item in trajectories)

    split_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "strip_mode": strip_mode,
        "action_eps": action_eps,
        "hand_action_source": hand_action_source,
        "image_keys": image_keys,
        "cleaning_action_source": "raw_action",
        "cleaning_action_scope": clean_action_scope,
        "total_success": len(success),
        "total_failure": len(failure),
        "n_train": len(train_files),
        "n_test": len(test_files),
        "train_frames": train_frames,
        "test_frames": test_frames,
        "failure_frames": failure_frames,
        "frames_before_cleaning": total_frames_before,
        "frames_after_cleaning": total_frames_after,
        "blank_frames_removed": blank_frames_removed,
        "inputs": [str(path) for path in sorted(input_dir.glob("*.pkl"))],
        "train_files": sorted(train_files),
        "test_files": sorted(test_files),
    }
    with (success_dir / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    metadata = {
        "trajectory_format": "pt",
        "size": len(trajectories),
        "total_samples": total_frames_after,
        "trajectory_counter": len(trajectories),
        "seed": seed,
        "inputs": [str(path) for path in sorted(input_dir.glob("*.pkl"))],
        "success": len(success),
        "failure": len(failure),
        "blank_frames_removed": blank_frames_removed,
        "hand_action_source": hand_action_source,
        "image_keys": image_keys,
        "cleaning_action_source": "raw_action",
        "cleaning_action_scope": clean_action_scope,
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    with (output_dir / "conversion_manifest.json").open("w") as f:
        json.dump(sorted(manifest, key=lambda item: item["trajectory_index"]), f, indent=2)

    return {
        "success": len(success),
        "failure": len(failure),
        "train": len(train_files),
        "test": len(test_files),
        "frames_before": total_frames_before,
        "frames_after": total_frames_after,
        "blank_removed": blank_frames_removed,
    }


def trajectory_manifest_entry(
    trajectory: ConvertedTrajectory,
    filename: str,
    relative_dir: str,
) -> dict[str, Any]:
    return {
        "trajectory_index": trajectory.index,
        "filename": filename,
        "relative_dir": relative_dir,
        "source_file": trajectory.source_file,
        "source_episode_index": trajectory.source_episode_index,
        "success": trajectory.success,
        "frames_before_cleaning": trajectory.frames_before_cleaning,
        "frames_after_cleaning": trajectory.frames_after_cleaning,
        "blank_frames_removed": trajectory.blank_frames_removed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/example_task_raw"),
        help="Directory containing flattened .pkl demo files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output demos directory. Defaults to <input-dir>/demos.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of successful trajectories saved under success/train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Train/test split seed.")
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=100,
        help="Value stored in each output trajectory's max_episode_length field.",
    )
    parser.add_argument(
        "--action-eps",
        type=float,
        default=5e-3,
        help=(
            "Frames with selected raw action dims L2 norm <= this value are blank. "
            "See --clean-action-scope."
        ),
    )
    parser.add_argument(
        "--clean-action-scope",
        choices=["all", "wrist", "hand"],
        default="all",
        help=(
            "Raw action dimensions used for blank-frame detection: all=0:12, "
            "wrist=0:6, hand=6:12."
        ),
    )
    parser.add_argument(
        "--hand-action-source",
        choices=["gripper_pose", "state", "raw_action"],
        default="gripper_pose",
        help=(
            "Source for output actions[:, 6:12]. gripper_pose/state write absolute hand pose; "
            "raw_action keeps the pickle's original hand control values."
        ),
    )
    parser.add_argument(
        "--strip-mode",
        choices=["frame_norm", "head", "both", "none"],
        default="frame_norm",
        help=(
            "Success cleaning mode: frame_norm removes every blank frame; "
            "head removes only leading blanks; both trims leading/trailing blanks; "
            "none disables cleaning."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists and is non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().absolute()
    output_dir = (args.output_dir or (input_dir / "demos")).expanduser().absolute()

    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input directory not found: {input_dir}")
    if not (0.0 <= args.train_ratio <= 1.0):
        raise SystemExit("ERROR: --train-ratio must be between 0 and 1")
    if args.action_eps < 0:
        raise SystemExit("ERROR: --action-eps must be non-negative")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(
        f"Hand action source: {args.hand_action_source}; "
        f"cleaning source: raw_action[{args.clean_action_scope}]; "
        f"strip_mode={args.strip_mode}, action_eps={args.action_eps:g}"
    )

    trajectories = convert_all(
        input_dir=input_dir,
        max_episode_length=args.max_episode_length,
        action_eps=args.action_eps,
        strip_mode=args.strip_mode,
        clean_action_scope=args.clean_action_scope,
        hand_action_source=args.hand_action_source,
    )
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    summary = save_outputs(
        trajectories=trajectories,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        input_dir=input_dir,
        action_eps=args.action_eps,
        strip_mode=args.strip_mode,
        clean_action_scope=args.clean_action_scope,
        hand_action_source=args.hand_action_source,
    )

    print()
    print("Done.")
    print(f"  trajectories: {len(trajectories)}")
    print(f"  success:      {summary['success']} ({summary['train']} train / {summary['test']} test)")
    print(f"  failure:      {summary['failure']}")
    print(f"  frames:       {summary['frames_before']} -> {summary['frames_after']}")
    print(f"  blank removed:{summary['blank_removed']}")
    print(f"  metadata:     {output_dir / 'metadata.json'}")
    print(f"  split info:   {output_dir / 'success' / 'split_info.json'}")


if __name__ == "__main__":
    main()

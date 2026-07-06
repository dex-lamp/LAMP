"""Convert absolute-pose `actions` to delta-pose `actions` in trajectory .pt files.

Replaces the `actions` and `forward_inputs.action` tensors in every
`trajectory_*_demo_expert.pt` under --input-dir with their delta-action
equivalent, then writes the modified files into --output-dir, preserving
the directory layout. All other keys (observations, rewards, dones, ...)
are copied through unchanged. **Key names are not changed** — only the
underlying tensor values are replaced.

Delta convention (length-preserving, so obs and actions stay aligned):
    --method forward  (default): delta[t] = a[t+1] - a[t]; delta[T-1] = 0
    --method backward          : delta[t] = a[t] - a[t-1]; delta[0]   = 0

Why zero-pad the boundary instead of slicing? Slicing to T-1 would force
us to also slice every observation / reward / flag tensor; zero-padding
keeps everything aligned and lets downstream code stay unchanged.

Examples
--------
# Build a delta version of an absolute-action dataset:
python3 scripts/convert_to_delta_action.py \
    --input-dir  data/example_task/demos \
    --output-dir data/example_task/demos_delta \
    --copy-other-files

# Write a corrected sibling directory:
python3 scripts/convert_to_delta_action.py \
    --input-dir  data/example_task_delta/success/success \
    --output-dir data/example_task_delta_fixed/success/success \
    --copy-other-files
"""

import argparse
import shutil
from pathlib import Path

import torch


# ───────────────────────────── Core conversion ─────────────────────────────

def to_delta_actions(actions: torch.Tensor, method: str = "forward") -> torch.Tensor:
    """Convert an absolute action tensor to a delta action tensor.

    Length-preserving with a zero-padded boundary frame so that downstream
    tensors (obs, rewards, ...) remain aligned without needing changes.

    Args:
        actions: tensor of shape (T, ..., D)  — typically (T, 1, 12).
        method: 'forward'  → delta[t] = a[t+1] - a[t], delta[-1] = 0.
                'backward' → delta[t] = a[t] - a[t-1], delta[0]  = 0.
    """
    if actions.shape[0] < 2:
        return torch.zeros_like(actions)
    delta = torch.zeros_like(actions)
    if method == "forward":
        delta[:-1] = actions[1:] - actions[:-1]
    elif method == "backward":
        delta[1:] = actions[1:] - actions[:-1]
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return delta


def convert_trajectory_file(in_path: Path, out_path: Path, method: str) -> None:
    """Load one .pt trajectory, replace its actions in-memory, write to out_path."""
    data = torch.load(in_path, map_location="cpu", weights_only=False)

    if "actions" not in data:
        raise KeyError(f"{in_path} has no 'actions' key — got {list(data.keys())}")

    delta = to_delta_actions(data["actions"], method)
    data["actions"] = delta

    # The README states `forward_inputs["action"]` is a copy of `actions`,
    # so keep them consistent after conversion.
    fi = data.get("forward_inputs")
    if isinstance(fi, dict) and "action" in fi:
        fi["action"] = delta.clone()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)


def copy_aux_files(in_dir: Path, out_dir: Path) -> int:
    """Copy non-trajectory files (split_info.json, metadata.json, logs, ...) verbatim."""
    count = 0
    for src in in_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.name.startswith("trajectory_") and src.suffix == ".pt":
            continue  # already handled by the conversion pass
        dst = out_dir / src.relative_to(in_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


# ───────────────────────────── CLI ─────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Source directory containing trajectory_*_demo_expert.pt files (recursive).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Destination directory; will mirror the input structure.",
    )
    parser.add_argument(
        "--method", choices=["forward", "backward"], default="forward",
        help="Delta convention. forward: a[t+1]-a[t]; backward: a[t]-a[t-1]. Default: forward.",
    )
    parser.add_argument(
        "--copy-other-files", action="store_true",
        help="Also copy non-trajectory files (split_info.json, metadata.json, ...).",
    )
    args = parser.parse_args()

    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()

    if in_dir == out_dir:
        raise SystemExit("ERROR: --input-dir and --output-dir must differ (refusing to overwrite originals).")
    if not in_dir.is_dir():
        raise SystemExit(f"ERROR: input dir not found: {in_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"WARNING: output dir already exists and is non-empty: {out_dir}")
        print("         existing files with the same name will be OVERWRITTEN.")

    pt_files = sorted(in_dir.rglob("trajectory_*_demo_expert.pt"))
    if not pt_files:
        raise SystemExit(f"ERROR: no trajectory_*_demo_expert.pt files under {in_dir}")

    print(f"Found {len(pt_files)} trajectory files under {in_dir}")
    print(f"Writing converted ({args.method}) files to {out_dir}")

    for i, src in enumerate(pt_files, 1):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        convert_trajectory_file(src, dst, args.method)
        if i % 20 == 0 or i == len(pt_files):
            print(f"  [{i:>4}/{len(pt_files)}] {rel}")

    if args.copy_other_files:
        n_aux = copy_aux_files(in_dir, out_dir)
        print(f"Copied {n_aux} aux files (e.g. split_info.json, metadata.json)")

    print(f"\nDone. {len(pt_files)} trajectories → {out_dir}")


if __name__ == "__main__":
    main()

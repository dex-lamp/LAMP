"""Fit a raw-centered PCA baseline for single-frame hand6 actions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PCA_ROOT = Path(__file__).resolve().parents[1]
if str(PCA_ROOT) not in sys.path:
    sys.path.insert(0, str(PCA_ROOT))

from model.data_io import HAND_SLICE_DESCRIPTION, load_hand_action_dataset  # noqa: E402
from model.pca_hand import ACTION_DIM, evaluate_dims, fit_pca  # noqa: E402


DEFAULT_DATA_DIR = "data/example_task/demos"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA over raw single-frame hand6 actions")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Demos root containing success/{train,test}, or a direct trajectory directory.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/hand_pca", help="Directory for PCA outputs.")
    parser.add_argument("--save-plot", action="store_true", help="Save reconstruction_loss.png.")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def save_loss_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dim",
        "recon_mse",
        "unexplained_variance",
        "unexplained_variance_ratio",
        "cumulative_explained_variance_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not available; skipping plot")
        return

    dims = [row["dim"] for row in rows]
    recon_mse = [row["recon_mse"] for row in rows]
    unexplained = [row["unexplained_variance_ratio"] for row in rows]

    fig, ax_mse = plt.subplots(figsize=(7, 4))
    ax_ratio = ax_mse.twinx()
    ax_mse.plot(dims, recon_mse, "o-", color="royalblue", label="Reconstruction MSE")
    ax_ratio.plot(dims, unexplained, "s--", color="darkorange", label="Unexplained variance ratio")
    ax_mse.set_xlabel("PCA dimensions")
    ax_mse.set_ylabel("Reconstruction MSE")
    ax_ratio.set_ylabel("Unexplained variance ratio")
    ax_mse.set_xticks(dims)
    ax_mse.grid(True, alpha=0.3)
    lines, labels = ax_mse.get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    ax_mse.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()

    actions, data_metadata = load_hand_action_dataset(data_dir)
    pca = fit_pca(actions)
    rows = evaluate_dims(actions, pca["mean"], pca["components"], pca["explained_variance"])

    dims = np.arange(1, ACTION_DIM + 1, dtype=np.int32)
    projection_matrices = np.stack([row["projection_matrix"] for row in rows], axis=0)
    recon_mse_by_dim = np.asarray([row["recon_mse"] for row in rows], dtype=np.float64)
    unexplained_ratio_by_dim = np.asarray(
        [row["unexplained_variance_ratio"] for row in rows], dtype=np.float64
    )
    cumulative_ratio_by_dim = np.asarray(
        [row["cumulative_explained_variance_ratio"] for row in rows], dtype=np.float64
    )

    model_path = output_dir / "pca_model.npz"
    np.savez_compressed(
        model_path,
        mean=pca["mean"],
        components=pca["components"],
        explained_variance=pca["explained_variance"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        cumulative_explained_variance_ratio=np.cumsum(pca["explained_variance_ratio"]),
        projection_matrices=projection_matrices,
        dims=dims,
        recon_mse_by_dim=recon_mse_by_dim,
        unexplained_variance_ratio_by_dim=unexplained_ratio_by_dim,
        cumulative_explained_variance_ratio_by_dim=cumulative_ratio_by_dim,
        num_samples=np.asarray(pca["num_samples"], dtype=np.int64),
        action_dim=np.asarray(ACTION_DIM, dtype=np.int64),
    )

    csv_path = output_dir / "loss_by_dim.csv"
    save_loss_csv(rows, csv_path)

    plot_path = None
    if args.save_plot:
        plot_path = output_dir / "reconstruction_loss.png"
        save_plot(rows, plot_path)

    metrics_for_json = []
    for row in rows:
        item = dict(row)
        item.pop("projection_matrix")
        metrics_for_json.append(item)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir.resolve()),
        "input": {
            "hand_slice": HAND_SLICE_DESCRIPTION,
            "action_semantics": "raw absolute hand pose",
            "centering": "subtract per-dimension mean",
            "scaling": "none",
        },
        "data": data_metadata,
        "pca": {
            "num_samples": int(pca["num_samples"]),
            "action_dim": ACTION_DIM,
            "mean": pca["mean"],
            "explained_variance": pca["explained_variance"],
            "explained_variance_ratio": pca["explained_variance_ratio"],
            "total_variance": pca["total_variance"],
        },
        "metrics_by_dim": metrics_for_json,
        "outputs": {
            "pca_model": str(model_path),
            "loss_by_dim": str(csv_path),
            "reconstruction_loss_plot": str(plot_path) if plot_path is not None else None,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)

    print(f"Loaded {data_metadata['num_trajectories']} trajectories, {actions.shape[0]} hand6 samples")
    for source in data_metadata["sources"]:
        print(f"  {source['label']}: {source['num_trajectories']} trajectories from {source['path']}")
    print(f"Saved PCA model: {model_path}")
    print(f"Saved metrics:   {csv_path}")
    print(f"Saved summary:   {summary_path}")
    if plot_path is not None:
        print(f"Saved plot:      {plot_path}")
    print("\nLoss by dimension:")
    print("dim  recon_mse    unexplained_var_ratio")
    for row in rows:
        print(f"{row['dim']:>3d}  {row['recon_mse']:.8f}  {row['unexplained_variance_ratio']:.8f}")


if __name__ == "__main__":
    main()

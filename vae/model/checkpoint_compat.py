"""Utilities for JAX checkpoint persistence."""

import glob
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def latest_jax_checkpoint(checkpoint_dir: str) -> Optional[str]:
    latest_path = Path(checkpoint_dir) / "checkpoint.pkl"
    if latest_path.exists():
        return str(latest_path)
    candidates = glob.glob(os.path.join(str(checkpoint_dir), "checkpoint_*.pkl"))
    candidates.sort(key=lambda path: int(Path(path).stem.split("_")[-1]))
    return candidates[-1] if candidates else None


def save_jax_checkpoint(checkpoint_dir: str, payload: Dict[str, Any], step: int) -> str:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["step"] = int(step)
    step_path = checkpoint_path / f"checkpoint_{step}.pkl"
    latest_path = checkpoint_path / "checkpoint.pkl"
    with step_path.open("wb") as f:
        pickle.dump(payload, f)
    with latest_path.open("wb") as f:
        pickle.dump(payload, f)
    return str(step_path)


def restore_jax_checkpoint(checkpoint_dir: str, step: Optional[int] = None) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    if step is None:
        latest = latest_jax_checkpoint(str(checkpoint_path))
        if latest is None:
            raise FileNotFoundError(f"No JAX checkpoint found in {checkpoint_dir}")
        ckpt_file = Path(latest)
    else:
        ckpt_file = checkpoint_path / f"checkpoint_{step}.pkl"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"JAX checkpoint not found: {ckpt_file}")
    with ckpt_file.open("rb") as f:
        return pickle.load(f)

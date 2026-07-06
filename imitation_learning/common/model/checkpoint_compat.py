"""Checkpoint helpers for integrated JAX behavior_clone checkpoints."""

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


_BC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_VAE_ROOT = os.path.join(_PROJ_ROOT, "vae")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vae_compat_mod = _load_module(
    os.path.join(_VAE_ROOT, "model/checkpoint_compat.py"),
    "vae_checkpoint_compat_behavior_clone",
)
latest_jax_checkpoint = _vae_compat_mod.latest_jax_checkpoint
save_jax_checkpoint = _vae_compat_mod.save_jax_checkpoint
restore_jax_checkpoint = _vae_compat_mod.restore_jax_checkpoint


def _to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=True)
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(tensor, dtype=np.float32).copy()


def _torch_conv_to_flax(weight: Any) -> np.ndarray:
    return np.transpose(_to_numpy(weight), (2, 3, 1, 0)).copy()


def _bn_to_flax(torch_state: Dict[str, torch.Tensor], prefix: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    params = {
        "scale": _to_numpy(torch_state[f"{prefix}.weight"]),
        "bias": _to_numpy(torch_state[f"{prefix}.bias"]),
    }
    batch_stats = {
        "mean": _to_numpy(torch_state[f"{prefix}.running_mean"]),
        "var": _to_numpy(torch_state[f"{prefix}.running_var"]),
    }
    return params, batch_stats


def load_resnet18_torch_state_dict(pretrained_path: str) -> Dict[str, torch.Tensor]:
    from transformers import ResNetModel

    model = ResNetModel.from_pretrained(pretrained_path)
    return model.state_dict()


def flax_resnet_variables_from_torch_state_dict(
    torch_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {
        "embedder": {
            "embedder": {
                "convolution": {
                    "kernel": _torch_conv_to_flax(torch_state["embedder.embedder.convolution.weight"]),
                },
            },
        },
        "encoder": {"stages": {}},
    }
    batch_stats: Dict[str, Any] = {
        "embedder": {"embedder": {}},
        "encoder": {"stages": {}},
    }

    bn_params, bn_stats = _bn_to_flax(torch_state, "embedder.embedder.normalization")
    params["embedder"]["embedder"]["normalization"] = bn_params
    batch_stats["embedder"]["embedder"]["normalization"] = bn_stats

    for stage_idx in range(4):
        stage_params = {"layers": {}}
        stage_batch_stats = {"layers": {}}
        for layer_idx in range(2):
            layer_params = {"layer": {}}
            layer_batch_stats = {"layer": {}}
            for block_idx in range(2):
                base = f"encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{block_idx}"
                bn_params, bn_stats = _bn_to_flax(torch_state, f"{base}.normalization")
                layer_params["layer"][f"layer_{block_idx}"] = {
                    "convolution": {
                        "kernel": _torch_conv_to_flax(torch_state[f"{base}.convolution.weight"]),
                    },
                    "normalization": bn_params,
                }
                layer_batch_stats["layer"][f"layer_{block_idx}"] = {"normalization": bn_stats}

            if stage_idx > 0 and layer_idx == 0:
                base = f"encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut"
                bn_params, bn_stats = _bn_to_flax(torch_state, f"{base}.normalization")
                layer_params["shortcut"] = {
                    "convolution": {
                        "kernel": _torch_conv_to_flax(torch_state[f"{base}.convolution.weight"]),
                    },
                    "normalization": bn_params,
                }
                layer_batch_stats["shortcut"] = {"normalization": bn_stats}

            stage_params["layers"][str(layer_idx)] = layer_params
            stage_batch_stats["layers"][str(layer_idx)] = layer_batch_stats

        params["encoder"]["stages"][str(stage_idx)] = stage_params
        batch_stats["encoder"]["stages"][str(stage_idx)] = stage_batch_stats

    return params, batch_stats


def load_resnet18_flax_variables(pretrained_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return flax_resnet_variables_from_torch_state_dict(load_resnet18_torch_state_dict(pretrained_path))


def resolve_jax_checkpoint_dir(root: str) -> str:
    root_path = Path(root)
    if (root_path / "checkpoint.pkl").exists():
        return str(root_path)
    if (root_path / "jax_checkpoints" / "checkpoint.pkl").exists():
        return str(root_path / "jax_checkpoints")
    raise FileNotFoundError(f"Could not resolve JAX checkpoint dir from: {root}")

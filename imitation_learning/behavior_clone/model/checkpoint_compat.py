"""Compatibility exports for shared imitation-learning checkpoint helpers."""

import importlib.util
import os


_BC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_COMMON_CKPT_PATH = os.path.join(_PROJ_ROOT, "imitation_learning/common/model/checkpoint_compat.py")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_module(_COMMON_CKPT_PATH, "il_common_checkpoint_compat_for_behavior_clone")

latest_jax_checkpoint = _common.latest_jax_checkpoint
save_jax_checkpoint = _common.save_jax_checkpoint
restore_jax_checkpoint = _common.restore_jax_checkpoint
load_resnet18_torch_state_dict = _common.load_resnet18_torch_state_dict
flax_resnet_variables_from_torch_state_dict = _common.flax_resnet_variables_from_torch_state_dict
load_resnet18_config = _common.load_resnet18_config
load_resnet18_flax_variables = _common.load_resnet18_flax_variables
resolve_jax_checkpoint_dir = _common.resolve_jax_checkpoint_dir


__all__ = [
    "latest_jax_checkpoint",
    "save_jax_checkpoint",
    "restore_jax_checkpoint",
    "load_resnet18_torch_state_dict",
    "flax_resnet_variables_from_torch_state_dict",
    "load_resnet18_config",
    "load_resnet18_flax_variables",
    "resolve_jax_checkpoint_dir",
]

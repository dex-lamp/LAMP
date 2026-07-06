"""Compatibility exports for the shared BC JAX policy implementation."""

import importlib.util
import os


_BC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_COMMON_POLICY_PATH = os.path.join(_PROJ_ROOT, "imitation_learning/common/model/policy_jax.py")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_module(_COMMON_POLICY_PATH, "il_common_policy_jax_for_behavior_clone")

Array = _common.Array
IMAGENET_MEAN = _common.IMAGENET_MEAN
IMAGENET_STD = _common.IMAGENET_STD
DEFAULT_HIL_SERL_ROOT = _common.DEFAULT_HIL_SERL_ROOT
HandActionVAE = _common.HandActionVAE
TorchDense = _common.TorchDense
StateEncoderMLP = _common.StateEncoderMLP
StateEncoderLinear64 = _common.StateEncoderLinear64
StateEncoderRaw = _common.StateEncoderRaw
build_state_encoder_module = _common.build_state_encoder_module
CoreActionHead = _common.CoreActionHead
ResNet18Backbone = _common.ResNet18Backbone
HiLSerlResNet10Backbone = _common.HiLSerlResNet10Backbone
BCPolicy = _common.BCPolicy
count_params = _common.count_params


__all__ = [
    "Array",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_HIL_SERL_ROOT",
    "HandActionVAE",
    "TorchDense",
    "StateEncoderMLP",
    "StateEncoderLinear64",
    "StateEncoderRaw",
    "build_state_encoder_module",
    "CoreActionHead",
    "ResNet18Backbone",
    "HiLSerlResNet10Backbone",
    "BCPolicy",
    "count_params",
]

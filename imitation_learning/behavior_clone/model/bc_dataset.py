"""Compatibility exports for the shared BC dataset implementation."""

import importlib.util
import os


_BC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ_ROOT = os.path.abspath(os.path.join(_BC_ROOT, "..", ".."))
_COMMON_DATASET_PATH = os.path.join(_PROJ_ROOT, "imitation_learning/common/model/bc_dataset.py")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_module(_COMMON_DATASET_PATH, "il_common_bc_dataset_for_behavior_clone")

compute_action_stats = _common.compute_action_stats
BCDataset = _common.BCDataset
_edge_random_crop_chw = _common._edge_random_crop_chw
_paired_edge_random_crop_chw = _common._paired_edge_random_crop_chw
_multi_edge_random_crop_chw = _common._multi_edge_random_crop_chw


__all__ = [
    "compute_action_stats",
    "BCDataset",
    "_edge_random_crop_chw",
    "_paired_edge_random_crop_chw",
    "_multi_edge_random_crop_chw",
]

"""Export a sorted decoded hand-action codebook from a JAX VQ-VAE checkpoint."""

import argparse
import importlib.util
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np


_VQVAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_VQVAE_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model_mod = _load("model/hand_vqvae.py", "hand_vqvae_export_jax")
_ckpt_mod = _load("model/checkpoint_compat.py", "vqvae_checkpoint_export")

HandVQVAE = _model_mod.HandVQVAE
resolve_jax_checkpoint_dir = _ckpt_mod.resolve_jax_checkpoint_dir
restore_jax_checkpoint = _ckpt_mod.restore_jax_checkpoint


def get_args():
    p = argparse.ArgumentParser(description="Export sorted hand codebook from JAX VQ-VAE")
    p.add_argument("--ckpt_dir", type=str, required=True, help="VQ-VAE output dir or jax_checkpoints dir")
    p.add_argument("--output", type=str, default=None, help="Output .npy path")
    p.add_argument("--sort_by", type=str, default="pca", choices=["pca", "grip"])
    p.add_argument("--clip", action="store_true", help="Additionally clip decoded actions to checkpoint train action min/max")
    return p.parse_args()


def enumerate_indices(codebook_size: int, num_layers: int) -> np.ndarray:
    total = codebook_size ** num_layers
    return np.stack(np.unravel_index(np.arange(total), [codebook_size] * num_layers), axis=-1).astype(np.int32)


def pca_projection(actions: np.ndarray) -> np.ndarray:
    centered = actions - actions.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[0]


def grip_projection(actions: np.ndarray) -> np.ndarray:
    weights = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    return actions @ weights


def main():
    args = get_args()
    ckpt_dir = resolve_jax_checkpoint_dir(args.ckpt_dir)
    payload = restore_jax_checkpoint(ckpt_dir)
    model_args = payload["model_args"]
    model = HandVQVAE(**model_args)

    indices = enumerate_indices(int(model_args["codebook_size"]), int(model_args["num_vq_layers"]))
    decoded = model.apply(
        {"params": payload["params"], "vq_state": payload["vq_state"]},
        jnp.asarray(indices),
        method=model.decode_indices,
    )
    actions = np.asarray(decoded, dtype=np.float32)
    if bool(model_args.get("normalize_actions", True)):
        actions = np.clip(actions, 0.0, 1.0)
    raw_actions = actions.copy()

    if args.clip:
        action_min = np.asarray(payload.get("action_min", np.zeros((actions.shape[-1],))), dtype=np.float32)
        action_max = np.asarray(payload.get("action_max", np.ones((actions.shape[-1],))), dtype=np.float32)
        actions = np.clip(actions, action_min, action_max)

    projection = pca_projection(actions) if args.sort_by == "pca" else grip_projection(actions)
    sorted_order = np.argsort(projection)
    sorted_actions = actions[sorted_order]

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(ckpt_dir), "sorted_codebook.npy")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, sorted_actions)

    meta_path = output.replace(".npy", "_meta.npz")
    np.savez(
        meta_path,
        sorted_actions=sorted_actions,
        raw_actions=raw_actions,
        sorted_order=sorted_order,
        original_indices=indices[sorted_order],
        projection=projection[sorted_order],
        sort_by=args.sort_by,
        codebook_size=int(model_args["codebook_size"]),
        num_vq_layers=int(model_args["num_vq_layers"]),
        dq_rise_export_fixed_layer_average=True,
        normalized_training_actions=bool(model_args.get("normalize_actions", True)),
    )

    print(f"Decoded codebook shape: {actions.shape}")
    print(f"Sorted codebook shape: {sorted_actions.shape}")
    print(f"Saved: {output}")
    print(f"Meta:  {meta_path}")


if __name__ == "__main__":
    main()

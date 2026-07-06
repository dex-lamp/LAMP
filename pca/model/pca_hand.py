"""Raw-centered PCA for single-frame 6D hand actions."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_DIM = 6


def _as_hand_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (N, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("actions must contain at least one sample")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contains non-finite values")
    return actions


def _fix_component_signs(components: np.ndarray) -> np.ndarray:
    components = components.copy()
    for idx in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[idx])))
        if components[idx, pivot] < 0:
            components[idx] *= -1.0
    return components


def fit_pca(actions: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Fit PCA on raw hand actions after subtracting the per-dimension mean."""

    actions = _as_hand_actions(actions)
    mean = actions.mean(axis=0)
    centered = actions - mean
    denom = max(actions.shape[0] - 1, 1)
    covariance = (centered.T @ centered) / float(denom)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    explained_variance = np.maximum(eigvals[order], 0.0)
    components = _fix_component_signs(eigvecs[:, order].T)

    total_variance = float(explained_variance.sum())
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    return {
        "mean": mean,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "total_variance": total_variance,
        "num_samples": int(actions.shape[0]),
        "action_dim": ACTION_DIM,
    }


def project(actions: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int) -> np.ndarray:
    """Encode hand actions into the first ``dim`` PCA coordinates."""

    actions = _as_hand_actions(actions)
    dim = int(dim)
    if dim < 1 or dim > ACTION_DIM:
        raise ValueError(f"dim must be in [1, {ACTION_DIM}], got {dim}")
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    return (actions - mean) @ components[:dim].T


def reconstruct(actions: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int) -> np.ndarray:
    """Project to ``dim`` PCA coordinates and decode back to hand6 space."""

    z = project(actions, mean, components, dim)
    return z @ np.asarray(components, dtype=np.float64)[: int(dim)] + np.asarray(mean, dtype=np.float64)


def projection_matrix(components: np.ndarray, dim: int) -> np.ndarray:
    """Return the centered-space reconstruction matrix for the first ``dim`` PCs."""

    dim = int(dim)
    if dim < 1 or dim > ACTION_DIM:
        raise ValueError(f"dim must be in [1, {ACTION_DIM}], got {dim}")
    basis = np.asarray(components, dtype=np.float64)[:dim]
    return basis.T @ basis


def evaluate_dims(
    actions: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate reconstruction and information loss for PCA dimensions 1..6."""

    actions = _as_hand_actions(actions)
    explained_variance = np.asarray(explained_variance, dtype=np.float64)
    if explained_variance.shape != (ACTION_DIM,):
        raise ValueError(f"explained_variance must have shape ({ACTION_DIM},), got {explained_variance.shape}")

    total_variance = float(explained_variance.sum())
    rows: list[dict[str, Any]] = []
    for dim in range(1, ACTION_DIM + 1):
        recon = reconstruct(actions, mean, components, dim)
        residual = actions - recon
        recon_mse = float(np.mean(np.square(residual)))
        unexplained_variance = float(explained_variance[dim:].sum())
        if total_variance > 0.0:
            unexplained_ratio = unexplained_variance / total_variance
            cumulative_ratio = float(explained_variance[:dim].sum() / total_variance)
        else:
            unexplained_ratio = 0.0
            cumulative_ratio = 0.0
        rows.append(
            {
                "dim": dim,
                "recon_mse": recon_mse,
                "unexplained_variance": unexplained_variance,
                "unexplained_variance_ratio": float(unexplained_ratio),
                "cumulative_explained_variance_ratio": cumulative_ratio,
                "projection_matrix": projection_matrix(components, dim),
            }
        )
    return rows


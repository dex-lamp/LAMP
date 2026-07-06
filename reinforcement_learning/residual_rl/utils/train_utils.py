"""Small training helpers needed by the public residual RL algorithms.

This file intentionally keeps only generic utilities.  The private robot
launcher version also contained logging, video, and download helpers; those are
not needed for the supplement algorithm package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pickle as pkl

import jax


def _unpack(batch):
    """Split stacked observation sequences into observation/next-observation trees.

    Some replay buffers store image stacks only in ``batch["observations"]`` and
    omit duplicated pixels in ``batch["next_observations"]``.  The SAC update
    code expects both trees to contain the same keys, so this helper reconstructs
    the shifted next-observation pixels on demand.
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def load_resnet10_params(
    agent,
    image_keys: Iterable[str] = ("image",),
    *,
    params_path: str | Path = "pretrained_models/resnet10_params.pkl",
):
    """Load local SERL ResNet-10 parameters into an agent's visual encoders.

    The public package does not download weights automatically.  Place
    ``resnet10_params.pkl`` at ``params_path`` or pass an explicit path.
    """

    params_path = Path(params_path).expanduser()
    if not params_path.exists():
        raise FileNotFoundError(
            f"ResNet-10 params not found at {params_path}. "
            "Download or provide the file explicitly before using pretrained encoders."
        )
    with params_path.open("rb") as f:
        encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params))
    print(f"Loaded {param_count / 1e6:.1f}M ResNet-10 parameters from {params_path}")

    new_params = agent.state.params
    for image_key in tuple(image_keys):
        new_encoder_params = new_params["modules_actor"]["encoder"][f"encoder_{image_key}"]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for key in new_encoder_params:
            if key in encoder_params:
                new_encoder_params[key] = encoder_params[key]

    return agent.replace(state=agent.state.replace(params=new_params))

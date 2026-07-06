"""
Dataset for Behavior Cloning over a frozen Hand-Action VAE.

NAMING CONVENTION (CRITICAL):
  In this dataset the field `actions` represents the robot's ABSOLUTE POSE at
  each timestep (12 dims = 6 arm + 6 hand). So `actions[t]` IS the robot
  state at time t. The BC policy predicts the NEXT pose `actions[t+1]`.

  ─────────────────────────────────────────────────────────
   time t            : observe state[t] = actions[t]
   BC predicts       : actions[t+1]   (the next absolute pose)
  ─────────────────────────────────────────────────────────

This matches the frozen VAE's training convention exactly: the VAE was
trained to predict `a_{t+1}` from window `[a_{t-7}..a_t]`, so we feed it
the same window and use its output as a prior over the next hand pose.

Each sample at trajectory frame t (0 <= t < T) yields:

  img_main      : (3, 128, 128)  float32 in [0, 1]
  img_extra     : (3, 128, 128)  float32 in [0, 1]
  extra_images  : (N, 3, 128, 128) float32 in [0, 1], optional extra views
                  after img_main/img_extra. Empty when using two image keys.
  state         : (12,)          float32, = actions[t] standardized by
                                  per-dim mean/std (z-score), then optionally
                                  masked for ablations. All policy modes consume
                                  the first 6 arm dims through an arm state
                                  encoder.
  past_hand_win : (8, 6)         float32, hand actions [a_{t-7}..a_t]
                                  inclusive of current frame, padded with
                                  hand[0] when t < 7. Matches the VAE's
                                  HandActionWindowDataset window EXACTLY.
                                  Training noise may be applied in raw hand
                                  pose space to this field.
  past_hand_win_raw:
                 (8, 6)          float32, clean raw hand-action history before
                                  training noise. Used by pca_raw.
  past_hand_state_win:
                 (8, 6)          float32, same hand-action history after
                                  z-score normalization with hand action
                                  mean/std, with a configurable std floor for
                                  near-constant hand dimensions. Used by
                                  mlp_direct, decoder_only, and vq_codebook.
                                  Training noise may be applied in normalized
                                  space to this field.
  gt_action     : (12,)          float32, target = actions[t+1] (raw, NOT
                                  normalized — the VAE decoder is trained
                                  in raw action space). For t == T-1 we use
                                  actions[T-1] (hold last pose).

IMPORTANT — hand history ownership:
  In VAE-prior mode, past_hand_win is consumed exclusively by the FROZEN VAE
  encoder to compute a prior (mu_p, log_var_p) over the next hand pose. The
  BC's job is to predict a small delta_z correction so that
  z_ctrl = mu_prior + delta_z. In mlp_direct, decoder_only, and vq_codebook
  modes, the trainable direct head consumes an encoded normalized
  past_hand_state_win chunk.
  In pca_raw mode, the trainable head consumes a PCA-encoded clean raw
  past_hand_win_raw chunk.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_HAND_HISTORY_STD_FLOOR = 0.02


def compute_action_stats(data_dir: str):
    """One pass over the train split: per-dim mean/std of 12-dim action vectors.

    Returns:
        mean: (12,) float32
        std:  (12,) float32  (clamped to >= 1e-6 to avoid divide-by-zero)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {data_dir}")

    chunks = []
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        chunks.append(data["actions"][:, 0, :].float())  # (T, 12)
    all_actions = torch.cat(chunks, dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp(min=1e-6)
    return mean, std


def _edge_random_crop_chw(img: torch.Tensor, crop_y: int, crop_x: int, padding: int) -> torch.Tensor:
    """Random-shift crop matching the edge-padded crop used during deployment."""
    if padding <= 0:
        return img
    _, height, width = img.shape
    padded = F.pad(
        img.unsqueeze(0),
        (padding, padding, padding, padding),
        mode="replicate",
    ).squeeze(0)
    return padded[:, crop_y : crop_y + height, crop_x : crop_x + width].contiguous()


def _paired_edge_random_crop_chw(
    img_main: torch.Tensor,
    img_extra: torch.Tensor,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same edge-padded random crop offset to both camera views."""
    if padding <= 0:
        return img_main, img_extra
    crop_from = torch.randint(0, 2 * padding + 1, (2,), dtype=torch.int64)
    crop_y, crop_x = int(crop_from[0]), int(crop_from[1])
    return (
        _edge_random_crop_chw(img_main, crop_y, crop_x, padding),
        _edge_random_crop_chw(img_extra, crop_y, crop_x, padding),
    )


def _multi_edge_random_crop_chw(
    images: tuple[torch.Tensor, ...],
    padding: int,
) -> tuple[torch.Tensor, ...]:
    """Apply the same edge-padded random crop offset to all camera views."""
    if padding <= 0 or not images:
        return images
    crop_from = torch.randint(0, 2 * padding + 1, (2,), dtype=torch.int64)
    crop_y, crop_x = int(crop_from[0]), int(crop_from[1])
    return tuple(
        _edge_random_crop_chw(image, crop_y, crop_x, padding)
        for image in images
    )


def _normalize_image_keys(image_keys) -> tuple[str, ...]:
    if isinstance(image_keys, str):
        keys = tuple(key.strip() for key in image_keys.replace(",", " ").split() if key.strip())
    else:
        keys = tuple(image_keys)
    if len(keys) < 2:
        raise ValueError(f"BCDataset requires at least two image keys, got {keys}")
    return keys


def _extra_view_images_for_key(data: dict, key: str) -> torch.Tensor:
    curr_obs = data["curr_obs"]
    if key == "global":
        return curr_obs["main_images"][:, 0]

    extra_view_images = curr_obs.get("extra_view_images")
    if extra_view_images is None:
        raise KeyError(f"curr_obs is missing extra_view_images for image key {key!r}")
    if extra_view_images.ndim != 6:
        raise ValueError(
            f"extra_view_images must have shape (T, B, V, H, W, C), got {tuple(extra_view_images.shape)}"
        )
    view_index_by_key = {
        "wrist": 0,
        "global_2": 1,
        "global2": 1,
    }
    if key not in view_index_by_key:
        raise KeyError(
            f"Unsupported image key {key!r}; supported keys are global, wrist, global_2."
        )
    view_index = view_index_by_key[key]
    if extra_view_images.shape[2] <= view_index:
        raise KeyError(
            f"Requested image key {key!r}, but extra_view_images only has "
            f"{extra_view_images.shape[2]} view(s)."
        )
    return extra_view_images[:, 0, view_index]


def _load_image_view(data: dict, key: str) -> torch.Tensor:
    images = _extra_view_images_for_key(data, key)
    return images.permute(0, 3, 1, 2).contiguous()


class BCDataset(Dataset):
    """BC dataset: per-step observation + next-action target.

    Args:
        data_dir:    directory with trajectory_*_demo_expert.pt files
        action_mean: (12,) per-dim mean for state standardization
        action_std:  (12,) per-dim std (already clamped)
        window_size: VAE prior window size (must match VAE training; default 8)
        enable_image_random_crop: if True, apply paired edge-padded random crop
                                  to img_main/img_extra in __getitem__.
        random_crop_padding: crop padding used when enable_image_random_crop=True.
    """

    def __init__(
        self,
        data_dir: str,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        window_size: int = 8,
        noise_std_hand: float = 0.1,
        noise_std_arm: float = 0.0,
        hand_codebook=None,
        image_keys=("global", "wrist"),
        enable_image_random_crop: bool = False,
        random_crop_padding: int = 4,
        hand_history_std_floor: float = DEFAULT_HAND_HISTORY_STD_FLOOR,
    ):
        data_dir = Path(data_dir)
        traj_files = sorted(data_dir.glob("trajectory_*_demo_expert.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files in {data_dir}")

        self.window_size = window_size
        self.image_keys = _normalize_image_keys(image_keys)
        self.num_image_views = len(self.image_keys)
        self.action_mean = action_mean.clone().float()
        self.action_std = action_std.clone().float()
        if hand_history_std_floor < 0:
            raise ValueError(f"hand_history_std_floor must be >= 0, got {hand_history_std_floor}")
        self.hand_history_std_floor = float(hand_history_std_floor)
        self.hand_history_std = torch.maximum(
            self.action_std[6:12],
            torch.full_like(self.action_std[6:12], max(self.hand_history_std_floor, 1e-6)),
        )
        self.noise_std_hand = float(noise_std_hand)
        self.noise_std_arm = float(noise_std_arm)
        if random_crop_padding < 0:
            raise ValueError(f"random_crop_padding must be >= 0, got {random_crop_padding}")
        self.enable_image_random_crop = bool(enable_image_random_crop)
        self.random_crop_padding = int(random_crop_padding)
        self.hand_codebook = None
        if hand_codebook is not None:
            self.hand_codebook = torch.as_tensor(hand_codebook, dtype=torch.float32)
            if self.hand_codebook.ndim != 2 or self.hand_codebook.shape[1] != 6:
                raise ValueError(f"hand_codebook must have shape (N, 6), got {tuple(self.hand_codebook.shape)}")

        # Per-trajectory tensors (preloaded into RAM)
        self.actions = []     # list of (T, 12) float32
        self.image_views = []  # list of tuple[(T, 3, 128, 128) uint8, ...]
        self.imgs_main = []    # list of (T, 3, 128, 128) uint8
        self.imgs_extra = []   # list of (T, 3, 128, 128) uint8

        # Flat sample index: list of (traj_idx, t)
        self.samples = []

        for traj_idx, f in enumerate(traj_files):
            data = torch.load(f, map_location="cpu", weights_only=False)

            actions = data["actions"][:, 0, :].float()                # (T, 12)
            image_views = tuple(_load_image_view(data, key) for key in self.image_keys)

            self.actions.append(actions)
            self.image_views.append(image_views)
            self.imgs_main.append(image_views[0])
            self.imgs_extra.append(image_views[1])
            for t in range(actions.shape[0]):
                self.samples.append((traj_idx, t))

        n_frames = len(self.samples)
        bytes_imgs = sum(
            sum(view.numel() for view in image_views)
            for image_views in self.image_views
        )
        print(
            f"BCDataset: {len(traj_files)} trajectories, {n_frames} frames "
            f"from {data_dir}  (image RAM ~{bytes_imgs / 1e6:.0f} MB uint8, "
            f"image_keys={self.image_keys}, "
            f"enable_image_random_crop={self.enable_image_random_crop}, "
            f"random_crop_padding={self.random_crop_padding}, "
            f"hand_history_std_floor={self.hand_history_std_floor:g})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        traj_idx, t = self.samples[idx]

        actions = self.actions[traj_idx]      # (T, 12)
        image_views = self.image_views[traj_idx]
        T = actions.shape[0]

        # ── BC trainable inputs at time t ──
        images = tuple(view[t].float() / 255.0 for view in image_views)
        if self.enable_image_random_crop:
            images = _multi_edge_random_crop_chw(images, self.random_crop_padding)
        img_main, img_extra = images[:2]
        if len(images) > 2:
            extra_images = torch.stack(images[2:], dim=0)
        else:
            extra_images = torch.empty((0,) + tuple(img_main.shape), dtype=img_main.dtype)
        # state = actions[t] (current absolute pose), z-score normalized and
        # optionally ablated to test whether timing is leaking through state.
        state = (actions[t] - self.action_mean) / self.action_std  # (12,)
        # Optional Gaussian noise on arm state (training only — set 0 for test)
        if self.noise_std_arm > 0:
            state = state.clone()
            state[:6] = state[:6] + torch.randn(6) * self.noise_std_arm

        # ── Frozen VAE input (NOT a BC trainable input) ──
        # past_hand_win = 8 frames [a_{t-7}..a_t] inclusive of current frame.
        # Padded with hand[0] when t < window_size - 1. Matches VAE training.
        hand = actions[:, 6:12]
        start = t - self.window_size + 1      # = t - 7
        if start < 0:
            pad_len = -start
            past_hand_win = torch.cat(
                [hand[0:1].expand(pad_len, -1), hand[0:t + 1]],
                dim=0,
            )
        else:
            past_hand_win = hand[start:t + 1]
        assert past_hand_win.shape == (self.window_size, 6), (
            f"past_hand_win shape {past_hand_win.shape} at traj={traj_idx} t={t}"
        )
        past_hand_win_raw = past_hand_win.clone()
        past_hand_state_win = (past_hand_win_raw - self.action_mean[6:12]) / self.hand_history_std
        # Optional Gaussian noise on hand history (training only — set 0 for test).
        # VAE consumes raw hand poses, while direct-head modes consume z-scored
        # hand history. Add noise in the space each branch actually sees; adding
        # raw 0.1 noise before z-scoring explodes near-constant hand dimensions.
        if self.noise_std_hand > 0:
            past_hand_win = past_hand_win_raw + torch.randn_like(past_hand_win_raw) * self.noise_std_hand
            past_hand_state_win = past_hand_state_win + torch.randn_like(past_hand_state_win) * self.noise_std_hand

        # ── Target: actions[t+1] (or actions[t] for last frame) ──
        if t + 1 < T:
            gt_action = actions[t + 1]
        else:
            gt_action = actions[t]

        ret = {
            "img_main": img_main,
            "img_extra": img_extra,
            "extra_images": extra_images,
            "state": state,
            "past_hand_win": past_hand_win,
            "past_hand_win_raw": past_hand_win_raw,
            "past_hand_state_win": past_hand_state_win,
            "gt_action": gt_action,
        }
        if self.hand_codebook is not None:
            hand = gt_action[6:12]
            distances = torch.sum((self.hand_codebook - hand.unsqueeze(0)) ** 2, dim=-1)
            gt_hand_index = torch.argmin(distances).long()
            denom = max(int(self.hand_codebook.shape[0]) - 1, 1)
            gt_hand_index_norm = gt_hand_index.float() / float(denom) * 2.0 - 1.0
            ret["gt_hand_index"] = gt_hand_index.float()
            ret["gt_hand_index_norm"] = gt_hand_index_norm.view(1)
            ret["gt_action_vq"] = torch.cat([gt_action[:6], gt_hand_index_norm.view(1)], dim=0)
        return ret

"""Observation context wrapper for BCVAE-based policies.

BCVAE checkpoints expect more than the standard SERL observation: they need the
previous 6D arm command, the current absolute 6D hand pose, and a fixed-length
history of absolute hand poses.  This module owns that online state tracking.

The wrapper is intentionally orthogonal to robot control.  It does not change
the environment action, reward, or done logic; it only augments observations
with the BCVAE context keys consumed by `BCVAEPolicy`, `ResidualSACAgent`, and
`BCHeadTD3Agent`.
"""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


BC_POLICY_STATE_KEY = "bc_policy_state"
PAST_HAND_WIN_KEY = "past_hand_win"
CURRENT_HAND_ABS_KEY = "current_hand_abs"


def _maybe_unstack_state(obs_state: Any) -> np.ndarray:
    """Flatten a SERL `state` observation and remove a length-1 chunk dimension."""
    state = np.asarray(obs_state, dtype=np.float32)
    if state.ndim >= 2 and state.shape[0] == 1:
        state = state[0]
    return state.reshape(-1)


def get_current_hand_abs(
    obs: dict[str, Any],
    info: dict[str, Any] | None,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return the current absolute 6D hand pose.

    The preferred source is `info["original_state_obs"]["gripper_pose"]`, which
    `RelativeFrame` preserves before SERL flattening.  For dry runs, saved demos,
    or already-augmented observations, the function falls back to top-level
    `current_hand_abs`, `info["gripper_pose"]`, flattened `obs["state"]`, and
    finally the caller-provided fallback. In the expected observation layout,
    `gripper_pose` is the final proprio key, so the flattened fallback reads
    the last six dimensions.
    """
    info = info or {}
    original_state = info.get("original_state_obs", {})
    if isinstance(original_state, dict) and "gripper_pose" in original_state:
        hand = np.asarray(original_state["gripper_pose"], dtype=np.float32)
    elif CURRENT_HAND_ABS_KEY in obs:
        hand = np.asarray(obs[CURRENT_HAND_ABS_KEY], dtype=np.float32)
    elif "gripper_pose" in info:
        hand = np.asarray(info["gripper_pose"], dtype=np.float32)
    elif "state" in obs:
        state = _maybe_unstack_state(obs["state"])
        if state.size < 6:
            raise ValueError(f"Cannot infer 6D hand pose from state shape {state.shape}")
        hand = state[-6:]
    elif fallback is not None:
        hand = np.asarray(fallback, dtype=np.float32)
    else:
        raise KeyError("Could not find current 6D hand pose in info or observation.")

    hand = np.asarray(hand, dtype=np.float32).reshape(-1)
    if hand.shape != (6,):
        raise ValueError(f"Expected current hand pose shape (6,), got {hand.shape}")
    return np.clip(hand, 0.0, 1.0).astype(np.float32)


def make_bc_vae_observation_space(
    observation_space: gym.spaces.Dict, window_size: int
) -> gym.spaces.Dict:
    """Add BCVAE context keys to an existing SERL observation space."""
    spaces = dict(observation_space.spaces)
    spaces[BC_POLICY_STATE_KEY] = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(12,),
        dtype=np.float32,
    )
    spaces[PAST_HAND_WIN_KEY] = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(int(window_size), 6),
        dtype=np.float32,
    )
    spaces[CURRENT_HAND_ABS_KEY] = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(6,),
        dtype=np.float32,
    )
    return gym.spaces.Dict(spaces)


def augment_bc_vae_observation(
    obs: dict[str, Any],
    prev_arm_action: np.ndarray,
    current_hand_abs: np.ndarray,
    hand_history: deque[np.ndarray],
) -> dict[str, Any]:
    """Return `obs` with BCVAE policy context keys added or overwritten."""
    obs_aug = dict(obs)
    obs_aug[BC_POLICY_STATE_KEY] = np.concatenate(
        [prev_arm_action, current_hand_abs], axis=0
    ).astype(np.float32)
    obs_aug[PAST_HAND_WIN_KEY] = np.stack(list(hand_history), axis=0).astype(np.float32)
    obs_aug[CURRENT_HAND_ABS_KEY] = np.asarray(current_hand_abs, dtype=np.float32)
    return obs_aug


class BCVAEPolicyContext:
    """State machine that builds BCVAE context for online or offline transitions."""

    def __init__(self, window_size: int):
        self.window_size = int(window_size)
        self.prev_arm_action = np.zeros((6,), dtype=np.float32)
        self.current_hand_abs = np.zeros((6,), dtype=np.float32)
        self.hand_history = deque(maxlen=self.window_size)
        self.reset_history(self.current_hand_abs)

    def reset_history(self, hand_abs: np.ndarray) -> None:
        """Fill the hand-history window with the reset hand pose."""
        hand_abs = np.asarray(hand_abs, dtype=np.float32).reshape(6)
        self.hand_history = deque(
            [hand_abs.copy() for _ in range(self.window_size)],
            maxlen=self.window_size,
        )

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start a new episode and return the first augmented observation."""
        self.prev_arm_action = np.zeros((6,), dtype=np.float32)
        self.current_hand_abs = get_current_hand_abs(
            obs,
            info or {},
            fallback=self.current_hand_abs,
        )
        self.reset_history(self.current_hand_abs)
        return self.augment(obs)

    def augment(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Add the current context state to `obs` without advancing history."""
        return augment_bc_vae_observation(
            obs,
            self.prev_arm_action,
            self.current_hand_abs,
            self.hand_history,
        )

    def step(
        self,
        next_obs: dict[str, Any],
        next_info: dict[str, Any] | None,
        executed_action: np.ndarray,
    ) -> dict[str, Any]:
        """Advance context after one environment step.

        `executed_action` must be the real 12D action sent to the robot.  Actor
        loops should pass intervention actions here when a human override
        replaced the policy action.
        """
        executed_action = np.asarray(executed_action, dtype=np.float32).reshape(12)
        fallback_hand = np.clip(
            self.current_hand_abs + executed_action[6:12],
            0.0,
            1.0,
        ).astype(np.float32)
        self.current_hand_abs = get_current_hand_abs(
            next_obs,
            next_info or {},
            fallback=fallback_hand,
        )
        self.hand_history.append(self.current_hand_abs.copy())
        self.prev_arm_action = executed_action[:6].astype(np.float32)
        return self.augment(next_obs)


class BCVAEContextWrapper(gym.Wrapper):
    """Gym wrapper that appends BCVAE context to observations online.

    The wrapped environment must expose a 12D action space:
    `[arm_action_6, hand_delta_6]`.  The wrapper does not alter actions; it only
    observes the action actually executed so the next observation's context is
    consistent with the replay transition.
    """

    def __init__(self, env: gym.Env, window_size: int):
        super().__init__(env)
        self.context = BCVAEPolicyContext(window_size)
        self.observation_space = make_bc_vae_observation_space(
            self.env.observation_space,
            window_size,
        )
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.context.reset(obs, info), info

    def step(self, action):
        next_obs, reward, done, truncated, info = self.env.step(action)
        # Intervention wrappers annotate the real override action in `info`.
        # Use it for history, but leave the key intact so actor loops can still
        # store the correct action in replay and count intervention stats.
        executed_action = info.get("intervene_action", action)
        next_obs_aug = self.context.step(next_obs, info, executed_action)
        return next_obs_aug, reward, done, truncated, info

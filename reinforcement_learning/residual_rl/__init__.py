"""Residual RL package detached from robot interaction infrastructure."""

from reinforcement_learning.residual_rl.agents.continuous.residual_sac import ResidualSACAgent
from reinforcement_learning.residual_rl.utils.factory import make_residual_sac_pixel_agent

__all__ = ["ResidualSACAgent", "make_residual_sac_pixel_agent"]

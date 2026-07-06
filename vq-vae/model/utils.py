"""Utility functions for VQ-VAE training."""

import math

import numpy as np


def cosine_scheduler(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0):
    """
    Cosine learning rate scheduler based on steps.

    Args:
        base_value: Maximum learning rate value.
        final_value: Minimum learning rate value.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        start_warmup_value: Initial learning rate during warmup.

    Returns:
        np.ndarray of shape (total_steps,) with learning rate for each step.
    """
    warmup_steps = min(max(int(warmup_steps), 0), int(total_steps))
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        schedule = np.array([])
    else:
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / decay_steps))
            for i in range(decay_steps)
        ])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule

"""Utility functions for VAE training."""

import math

import numpy as np


def cosine_scheduler(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0):
    """Cosine LR scheduler with optional linear warmup."""
    warmup_steps = min(max(int(warmup_steps), 0), int(total_steps))
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps > 0:
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / decay_steps))
            for i in range(decay_steps)
        ])
    else:
        schedule = np.array([])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule


def beta_annealing_schedule(target_beta, total_steps, warmup_steps=2000):
    """Linear annealing of KL weight β from 0 to target_beta over warmup_steps."""
    warmup_steps = min(max(int(warmup_steps), 0), int(total_steps))
    schedule = np.zeros(total_steps)
    if warmup_steps > 0:
        schedule[:warmup_steps] = np.linspace(0, target_beta, warmup_steps)
        schedule[warmup_steps:] = target_beta
    else:
        schedule[:] = target_beta
    return schedule

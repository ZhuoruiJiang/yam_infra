"""Shared poses used by follower shutdown and policy rollout."""
import numpy as np


def shutdown_pose(state):
    """Zero the six joints of each arm while preserving measured grippers."""
    state = np.asarray(state)
    if state.ndim != 1 or not state.size or state.size % 7 or not np.isfinite(state).all():
        raise ValueError("Reset pose requires finite groups of six joints plus one gripper")
    target = np.zeros(state.shape, dtype=np.float32)
    target[6::7] = state[6::7]
    return target

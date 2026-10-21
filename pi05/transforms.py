"""Shared mapping. Export and live robot units/order stay unchanged."""
from dataclasses import dataclass
import numpy as np
from pi05.dataset import rgb_bytes

CAMERA_MAP = {"top": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}
JOINT_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


@dataclass(frozen=True)
class YAMInputs:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite state values in left/right order")
        result = {"state": state, "image": {target: rgb_bytes(data["images"][source])
                  for source, target in CAMERA_MAP.items()},
                  "image_mask": {key: np.True_ for key in CAMERA_MAP.values()},
                  "prompt": data["prompt"]}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 14 or not np.isfinite(actions).all():
                raise ValueError("Expected finite (horizon, 14) actions")
            result["actions"] = actions
        return result


@dataclass(frozen=True)
class YAMOutputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :14]}

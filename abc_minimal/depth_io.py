"""Read lossless exported depth and make a fixed-scale display preview."""

import json
import numpy as np


class DepthFrames:
    def __init__(self, path, num_steps):
        import h5py

        self.file = h5py.File(path, "r")
        try:
            self.cameras = json.loads(self.file.attrs["camera_order"])
            if len(self.file["timeline_ns"]) != num_steps:
                raise ValueError("Depth timeline length differs from episode")
        except Exception:
            self.close()
            raise

    def frame(self, index):
        panels = []
        for camera in self.cameras:
            group = self.file[camera]
            raw = group["frames"][index]
            meters = raw.astype(np.float32) * float(group["frames"].attrs["depth_scale_m"])
            # Fixed 0–1 m scale: near red, far blue; missing depth black.
            distance = np.clip(meters, 0, 1)
            rgb = np.stack([1 - distance, 1 - np.abs(2 * distance - 1), distance], axis=-1)
            rgb[(raw == 0) | (not bool(group["valid"][index]))] = 0
            # Match the ABC RGB preview size, nearest-neighbor for display only.
            y = np.linspace(0, raw.shape[0] - 1, 168).astype(int)
            x = np.linspace(0, raw.shape[1] - 1, 224).astype(int)
            panels.append((rgb[y[:, None], x[None, :]] * 255).astype(np.uint8))
        return np.concatenate(panels, axis=0)

    def close(self):
        self.file.close()

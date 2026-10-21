"""Read ABC exports without constructing ABC's processed EpisodeDataset."""
import json
import hashlib
from bisect import bisect_right
from pathlib import Path

import numpy as np

CAMERAS = ("top", "left", "right")
DIM = 14


def rgb_bytes(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three channels, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating RGB must be finite and in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class ABCDataset:
    def __init__(self, data_dir, horizon=50, prompt=None, episode_glob="*", decode=True):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon, self.prompt, self.decode = horizon, prompt, decode
        self.episodes, self.ends = [], []
        for path in sorted(Path(data_dir).glob(episode_glob)):
            if not (path / "episode_metadata.json").is_file():
                continue
            meta = json.loads((path / "episode_metadata.json").read_text())
            if meta.get("usable") is False:
                continue
            size = (path / "states_actions.bin").stat().st_size
            if size % (28 * 8):
                raise ValueError(f"Malformed state/action file: {path}")
            length = size // (28 * 8)
            if length != meta.get("num_frames", length) or length == 0:
                raise ValueError(f"Frame count mismatch or empty episode: {path}")
            if not np.isclose(meta.get("fps", 0), 30):
                raise ValueError(f"Expected 30 Hz export: {path}")
            if not set(CAMERAS).issubset(meta.get("cameras", [])):
                raise ValueError(f"Missing RGB cameras: {path}")
            if not (path / "combined_camera-images-rgb.mp4").is_file():
                raise FileNotFoundError(path / "combined_camera-images-rgb.mp4")
            self.episodes.append((path, length, meta))
            self.ends.append((self.ends[-1] if self.ends else 0) + length)
        if not self.episodes:
            raise ValueError(f"No matching exported episodes in {data_dir} ({episode_glob})")

    def __len__(self):
        return self.ends[-1]

    def require_training_data(self):
        for path, _, meta in self.episodes:
            task = str(meta.get("task_name", "")).lower().replace("_", " ")
            if "depth_test" in path.name or "recording_test" in path.name or task in ("depth test", "recording test"):
                raise ValueError("Test recordings are validation fixtures; supply actual training demonstrations")

    def provenance(self):
        return {"horizon": self.horizon, "prompt": self.prompt,
                "action_representation": "joint_delta_gripper_absolute",
                "episodes": [{"name": path.name, "frames": length,
                              "numeric_size": (path / "states_actions.bin").stat().st_size,
                              "numeric_mtime_ns": (path / "states_actions.bin").stat().st_mtime_ns,
                              "metadata_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()}
                             for path, length, meta in self.episodes]}

    def locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        return ep, index - (self.ends[ep - 1] if ep else 0)

    def numeric(self, index):
        ep, frame = self.locate(index)
        path, length, meta = self.episodes[ep]
        # ABC's low-level range reader: no normalization or masking.
        from abc_minimal.dataloader import read_state_action_rows
        from types import SimpleNamespace
        rows = read_state_action_rows(path, frame, min(length, frame + self.horizon),
                                      SimpleNamespace(state_dim=DIM, action_dim=DIM))
        if not np.isfinite(rows).all():
            raise ValueError(f"Nonfinite robot values: {path}:{frame}")
        actions = rows[:, DIM:].astype(np.float32)
        # Repeat final target at the tail; never read into another episode.
        actions = np.concatenate([actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)])
        prompt = self.prompt or meta.get("instruction") or meta.get("task_name")
        for entry in meta.get("prompt_timeline", []):
            if entry["frame"] <= frame and self.prompt is None:
                prompt = entry["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing task instruction: {path}")
        return {"state": rows[0, :DIM].astype(np.float32), "actions": actions, "prompt": prompt}

    def __getitem__(self, index):
        sample = self.numeric(index)
        if self.decode:
            from abc_minimal.dataloader import decode_frame
            ep, frame = self.locate(index)
            path, length, meta = self.episodes[ep]
            images = decode_frame(path, frame, length, meta["cameras"], CAMERAS)
            sample["images"] = {key: rgb_bytes(images[key]) for key in CAMERAS}
        return sample

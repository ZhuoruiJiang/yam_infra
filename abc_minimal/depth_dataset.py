"""Worker-local random access to native, lossless episode depth."""

import json
import os
from collections import OrderedDict

import numpy as np


class EpisodeDepthReader:
    """Keep a bounded cache of open sidecars; never share handles across workers."""

    def __init__(self, camera_keys, max_open_files=4):
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.camera_keys = tuple(camera_keys)
        self.max_open_files = max_open_files
        self._pid = os.getpid()
        self._files = OrderedDict()

    def __getstate__(self):
        # Spawn workers receive configuration only, even if the parent sampled.
        return {**self.__dict__, "_files": OrderedDict(), "_pid": None}

    def close(self):
        for file, _ in self._files.values():
            file.close()
        self._files.clear()

    def __del__(self):
        if hasattr(self, "_files"):
            self.close()

    def read(self, ep_dir, index, episode_length):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        key = (ep_dir, episode_length)
        if key not in self._files:
            self._files[key] = self._open(ep_dir, episode_length)
            if len(self._files) > self.max_open_files:
                file, _ = self._files.popitem(last=False)[1]
                file.close()
        self._files.move_to_end(key)
        _, cameras = self._files[key]
        depth, masks = {}, {}
        for camera, (frames, valid_frames, scale) in cameras.items():
            raw = frames[index]
            mask = (raw != 0) & bool(valid_frames[index])
            meters = raw.astype(np.float32) * scale
            meters[~mask] = 0
            depth[camera] = meters[None, :, :]
            masks[camera] = mask[None, :, :]
        return depth, masks

    def _open(self, ep_dir, episode_length):
        import h5py

        metadata = json.loads((ep_dir / "episode_metadata.json").read_text())
        path = ep_dir / metadata.get("depth_file", "depth.h5")
        if not path.is_file():
            raise FileNotFoundError(f"{ep_dir.name}: load_depth requires {path}")
        file = h5py.File(path, "r")
        try:
            if file["timeline_ns"].shape != (episode_length,):
                raise ValueError("depth timeline length differs from episode")
            cameras = {}
            for camera in self.camera_keys:
                if camera not in file:
                    raise ValueError(f"missing depth camera {camera!r}")
                frames = file[f"{camera}/frames"]
                valid = file[f"{camera}/valid"]
                scale = float(frames.attrs.get("depth_scale_m", 0))
                if (frames.dtype != np.uint16 or frames.ndim != 3
                        or frames.shape[0] != episode_length
                        or min(frames.shape[1:]) <= 0):
                    raise ValueError(f"{camera}: expected uint16 depth (N,H,W)")
                if valid.shape != (episode_length,) or valid.dtype != np.bool_:
                    raise ValueError(f"{camera}: expected boolean frame validity (N,)")
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"{camera}: invalid depth_scale_m")
                cameras[camera] = (frames, valid, scale)
            return file, cameras
        except Exception as exc:
            file.close()
            raise ValueError(f"{path}: {exc}") from exc

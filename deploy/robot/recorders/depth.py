"""Lossless native depth storage and per-frame camera metadata."""

import json
import numpy as np


def append_camera_metadata(f, name, stream, metadata):
    group = f.require_group(f"camera_metadata/{name}/{stream}")
    for key, dtype in (("frameset_number", "uint64"), ("frame_number", "uint64"),
                       ("device_timestamp_ms", "float64")):
        if key not in metadata:
            raise ValueError(f"Missing camera metadata: {key}")
        if key not in group:
            group.create_dataset(key, (0,), maxshape=(None,), dtype=dtype, chunks=True)
        ds = group[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = metadata[key]
    for key in ("camera_serial", "timestamp_domain"):
        if key in metadata:
            group.attrs[key] = metadata[key]
    if "intrinsics" in metadata:
        group.attrs["intrinsics_json"] = json.dumps(metadata["intrinsics"])
    if "depth_to_color" in metadata:
        group.attrs["depth_to_color_json"] = json.dumps(metadata["depth_to_color"])


def append_depth(f, name, image, timestamp, metadata):
    if image.dtype != np.uint16 or image.ndim != 2:
        raise ValueError("Depth must be a two-dimensional uint16 array")
    scale = float(metadata["depth_scale_m"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be finite and positive")
    key = f"depth_{name}"
    data = f["data"]
    timestamps = f["timestamps"]
    if key not in data:
        ds = data.create_dataset(key, (0, *image.shape), maxshape=(None, *image.shape),
                                 dtype="uint16", chunks=(1, *image.shape),
                                 compression="gzip", compression_opts=1, shuffle=True)
        ds.attrs["depth_scale_m"] = scale
        ds.attrs["invalid_value"] = 0
        ds.attrs["aligned_to_rgb"] = False
        timestamps.create_dataset(key, (0,), maxshape=(None,), dtype="uint64", chunks=True)
    ds = data[key]
    if ds.shape[1:] != image.shape or ds.attrs["depth_scale_m"] != scale:
        raise ValueError("Depth shape or scale changed during recording")
    ds.resize(ds.shape[0] + 1, axis=0)
    timestamps[key].resize(timestamps[key].shape[0] + 1, axis=0)
    ds[-1] = image
    timestamps[key][-1] = int(timestamp)
    append_camera_metadata(f, name, "depth", metadata)

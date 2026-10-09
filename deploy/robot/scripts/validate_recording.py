"""Validate saved teleop HDF5 files without cameras or arms."""

import argparse
import json
import h5py
import numpy as np


def validate(path, require_depth=False, cameras=("left", "right", "top"), robots=("left", "right")):
    errors, warnings, streams = [], [], {}
    with h5py.File(path, "r") as f:
        if "data" not in f or "timestamps" not in f:
            return {"errors": ["Missing data or timestamps group"], "warnings": [], "streams": {}}
        for name, ds in f["data"].items():
            streams[name] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
            if ds.shape[0] == 0:
                errors.append(f"{name}: empty stream")
            if name not in f["timestamps"]:
                errors.append(f"{name}: missing timestamps")
                continue
            ts = f["timestamps"][name][:]
            if ts.ndim != 1 or len(ts) != ds.shape[0]:
                errors.append(f"{name}: timestamp count/shape mismatch")
            if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
                errors.append(f"{name}: timestamps go backwards")
        for camera in cameras:
            if camera not in f["data"]:
                errors.append(f"{camera}: missing RGB")
            else:
                ds = f["data"][camera]
                if ds.ndim != 1 or h5py.check_vlen_dtype(ds.dtype) != np.dtype("uint8"):
                    errors.append(f"{camera}: expected variable-length JPEG bytes")
                else:
                    for payload in ds:
                        if len(payload) < 4 or bytes(payload[:2]) != b'\xff\xd8' or bytes(payload[-2:]) != b'\xff\xd9':
                            errors.append(f"{camera}: invalid JPEG markers")
                            break
            key = f"depth_{camera}"
            if key not in f["data"]:
                (errors if require_depth else warnings).append(f"{camera}: no depth")
                continue
            depth = f["data"][key]
            scale = float(depth.attrs.get("depth_scale_m", 0))
            if depth.ndim != 3 or depth.dtype != np.uint16 or not np.isfinite(scale) or scale <= 0:
                errors.append(f"{camera}: invalid depth layout or scale")
            pair_ids = {}
            for stream, data_key in (("rgb", camera), ("depth", key)):
                if data_key not in f["data"]:
                    continue
                group = f.get(f"camera_metadata/{camera}/{stream}")
                if group is None:
                    errors.append(f"{camera}/{stream}: missing metadata")
                    continue
                for field in ("frameset_number", "frame_number", "device_timestamp_ms"):
                    if field not in group or len(group[field]) != f["data"][data_key].shape[0]:
                        errors.append(f"{camera}/{stream}: metadata count mismatch for {field}")
                if "frameset_number" in group:
                    pair_ids[stream] = set(group["frameset_number"][:].tolist())
                if "intrinsics_json" not in group.attrs:
                    warnings.append(f"{camera}/{stream}: no intrinsics")
            if len(pair_ids) == 2:
                unmatched = len(pair_ids["rgb"] ^ pair_ids["depth"])
                if unmatched:
                    warnings.append(f"{camera}: {unmatched} unpaired RGB/depth frameset IDs")
        for robot in robots:
            counts = []
            for prefix, width in (("q", 6), ("q_gripper", 1), ("q_vel", 7), ("q_eff", 7), ("q_des", 7)):
                key = f"{prefix}_{robot}"
                if key not in f["data"]:
                    errors.append(f"{key}: missing robot stream")
                    continue
                ds = f["data"][key]
                counts.append(ds.shape[0])
                if ds.ndim != 2 or ds.shape[1] != width or not np.isfinite(ds[:]).all():
                    errors.append(f"{key}: invalid shape or nonfinite values")
            if len(set(counts)) > 1:
                errors.append(f"{robot}: robot stream counts differ")
        stages = f.get("stages")
        if stages is None:
            warnings.append("No stages group")
        elif not all(key in stages for key in ("index", "start_ns", "end_ns")):
            errors.append("Incomplete stage metadata")
        else:
            indices, starts, ends = (stages[key][:] for key in ("index", "start_ns", "end_ns"))
            if len(starts) != len(indices) or len(ends) != len(indices):
                errors.append("Stage lengths differ")
            elif not np.array_equal(indices, np.arange(1, len(indices) + 1)) or np.any(ends < starts) or np.any(starts[1:] != ends[:-1]):
                errors.append("Invalid stage order or boundaries")
        if bool(f.attrs.get("discarded", False)) and bool(f.attrs.get("usable", True)):
            errors.append("Discarded recording is marked usable")
    return {"errors": errors, "warnings": warnings, "streams": streams}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--require-depth", action="store_true")
    args = parser.parse_args()
    result = validate(args.path, args.require_depth)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))


if __name__ == "__main__":
    main()

"""Smoke-test real RGB/depth decoding and batching without loading a model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/exports/depth_test"),
                        help="Export root containing train/ and norm_stats.json")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    try:
        import h5py
        import numpy as np
        import torch
        import torchcodec  # noqa: F401 - check the actual decoder dependency
    except ImportError as exc:
        raise SystemExit(f"Missing loader dependency: {exc}. Run in the training environment.") from exc

    from functools import partial
    from torch.utils.data import DataLoader, Subset
    from abc_minimal.config import DiTConfig
    from abc_minimal.dataloader import EpisodeDataset, collate
    from abc_minimal.preprocess import load_norm_stats

    config = DiTConfig()
    stats = load_norm_stats(args.data_root / "norm_stats.json")
    kwargs = dict(data_dir=args.data_root / "train", norm_stats=stats,
                  train=False, default_task_name="test", mask_state_ratio=0,
                  model_config=config)
    enabled = EpisodeDataset(**kwargs, load_depth=True)
    disabled = EpisodeDataset(**kwargs, load_depth=False)
    try:
        # The existing fixture contains older RGB-only episodes. Explicitly
        # select depth episodes here; production depth loading remains strict.
        indices, selected = [], []
        for i, (ep, length, usable, *_) in enumerate(enabled.episodes):
            import json
            meta = json.loads((ep / "episode_metadata.json").read_text())
            sidecar = ep / meta.get("depth_file", "depth.h5")
            if not sidecar.is_file():
                continue
            start = int(enabled.cum[i - 1]) if i else 0
            for frame in sorted({0, usable // 2, usable - 1}):
                indices.append(start + frame)
                selected.append((ep, sidecar, frame))
        if not indices:
            raise SystemExit("No usable depth episodes in train/.")

        rgb_only = disabled[indices[0]]
        assert "depth" not in rgb_only and "depth_valid" not in rgb_only
        rgb_batch = collate([rgb_only], config.camera_keys)
        assert "depth" not in rgb_batch and "depth_valid" not in rgb_batch

        loader = DataLoader(Subset(enabled, indices), batch_size=2,
                            num_workers=args.num_workers,
                            collate_fn=partial(collate, camera_keys=config.camera_keys))
        checked = 0
        for batch in loader:
            size = batch["state"].shape[0]
            assert batch["state"].shape == (size, config.state_dim)
            assert batch["actions"].shape == (size, config.chunk_length, config.action_dim)
            for j in range(size):
                ep, sidecar, frame = selected[checked + j]
                with h5py.File(sidecar, "r") as f:
                    for cam in config.camera_keys:
                        raw = f[f"{cam}/frames"][frame]
                        mask = (raw != 0) & bool(f[f"{cam}/valid"][frame])
                        expected = raw.astype(np.float32) * float(
                            f[f"{cam}/frames"].attrs["depth_scale_m"])
                        expected[~mask] = 0
                        depth = batch["depth"][cam][j]
                        valid = batch["depth_valid"][cam][j]
                        assert depth.dtype == torch.float32
                        assert valid.dtype == torch.bool
                        assert depth.shape == (1, *raw.shape)
                        np.testing.assert_allclose(depth.numpy()[0], expected)
                        np.testing.assert_array_equal(valid.numpy()[0], mask)
                        assert batch["images"][cam].shape == (size, 3, 224, 224)
            checked += size
        print(f"PASS: depth off adds no fields; depth on decoded and batched {checked} "
              f"real observations across {len(set(ep for ep, _, _ in selected))} episodes "
              f"with {args.num_workers} workers.")
        print("PASS: RGB/state/action shapes, native depth in meters, and validity masks.")
    finally:
        enabled.depth_reader.close()


if __name__ == "__main__":
    main()

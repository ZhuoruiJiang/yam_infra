"""Validate only exported depth_test episodes; never open depth sidecars."""
import argparse
import copy
from pathlib import Path
import numpy as np
from pi05.dataset import ABCDataset
from pi05.transforms import YAMInputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/exports/depth_test"))
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--numeric-only", action="store_true", help="Skip video decoding")
    p.add_argument("--openpi", action="store_true", help="Also validate OpenPI batch preprocessing")
    a = p.parse_args()
    if a.openpi and a.numeric_only:
        p.error("--openpi requires RGB decoding")
    ds = ABCDataset(a.data_root / "train", a.horizon, episode_glob="*depth_test*", decode=not a.numeric_only)
    checked = 0
    for ep, (path, length, _) in enumerate(ds.episodes):
        start = ds.ends[ep - 1] if ep else 0
        raw = np.fromfile(path / "states_actions.bin", dtype=np.float64).reshape(-1, 28)
        for frame in sorted({0, length // 2, length - 1}):
            item = ds[start + frame]
            np.testing.assert_allclose(item["state"], raw[frame, :14], rtol=1e-6, atol=1e-6)
            expected = raw[np.minimum(np.arange(frame, frame + a.horizon), length - 1), 14:]
            np.testing.assert_allclose(item["actions"], expected, rtol=1e-6, atol=1e-6)
            if not a.numeric_only:
                mapped = YAMInputs()(item)
                assert all(im.dtype == np.uint8 and im.shape[-1] == 3 for im in mapped["image"].values())
                assert "depth" not in item
                if a.openpi:
                    from openpi import transforms as ot
                    from openpi.models.pi0_config import Pi0Config
                    from openpi.training.config import ModelTransformFactory
                    from openpi.shared.normalize import NormStats
                    mask = ot.make_bool_mask(6, -1, 6, -1)
                    relative = ot.DeltaActions(mask)(copy.deepcopy(mapped))
                    restored = ot.AbsoluteActions(mask)(copy.deepcopy(relative))
                    np.testing.assert_allclose(restored["actions"], item["actions"], atol=1e-6)
                    # Identity test stats only, never written as training assets.
                    stats = {k: NormStats(mean=np.zeros(14), std=np.ones(14),
                             q01=-np.ones(14), q99=np.ones(14)) for k in ("state", "actions")}
                    transformed = ot.Normalize(stats, use_quantiles=True)(relative)
                    transformed = ot.compose(ModelTransformFactory()(Pi0Config(pi05=True, action_horizon=a.horizon)).inputs)(transformed)
                    assert transformed["actions"].shape == (a.horizon, 32)
                    assert transformed["state"].shape == (32,)
                    assert all(im.shape == (224, 224, 3) for im in transformed["image"].values())
            checked += 1
    print(f"PASS: {len(ds.episodes)} exported depth_test episodes, {len(ds)} frames, {checked} boundary/midpoint samples")
    print("RGB decoding: " + ("skipped" if a.numeric_only else "passed") + "; no depth loaded")


if __name__ == "__main__":
    main()

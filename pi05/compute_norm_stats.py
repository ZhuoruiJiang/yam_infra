"""Stream raw numeric chunks through the same joint delta transform as training."""
import argparse
import json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-tag", help="Select episode directories containing this literal tag; use the same tag for training")
    p.add_argument("--mode", choices=["full", "lora"], default="lora")
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--assets", help="Defaults to pi05/assets, or pi05/smoke_assets for smoke tests")
    p.add_argument("--smoke-test", action="store_true", help="Use test outputs; defaults to depth_test unless --data-tag is supplied")
    p.add_argument("--prompt")
    a = p.parse_args()
    a.assets = a.assets or ("pi05/smoke_assets" if a.smoke_test else "pi05/assets")
    from pathlib import Path
    from openpi.shared.normalize import RunningStats, save
    from pi05.dataset import ABCDataset, episode_pattern
    from pi05.training_config import make_config
    from pi05.transforms import JOINT_MASK
    ds = ABCDataset(Path(a.data_root) / "train", a.horizon, a.prompt,
                    episode_glob=episode_pattern(a.data_tag, a.smoke_test), decode=False)
    if not a.smoke_test:
        ds.require_training_data()
    stats = {k: RunningStats() for k in ("state", "actions")}
    for i in range(len(ds)):
        item = ds.numeric(i)
        actions = item["actions"].copy()
        actions[:, JOINT_MASK] -= item["state"][JOINT_MASK]
        stats["state"].update(item["state"][None].astype(np.float64))
        stats["actions"].update(actions.astype(np.float64))
    cfg = make_config(a.mode, a.horizon, a.assets)
    target = cfg.assets_dirs / "yam_abc"
    save(target, {key: value.get_statistics() for key, value in stats.items()})
    (target / "dataset.json").write_text(json.dumps(ds.provenance(), indent=2))
    print(f"Saved statistics for {len(ds)} observations to {target}")


if __name__ == "__main__":
    main()

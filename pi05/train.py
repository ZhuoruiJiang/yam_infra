"""Launch upstream OpenPI training with one scoped dataset-factory override."""
import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path, help="Export root containing train/")
    p.add_argument("--mode", choices=["full", "lora"], default="lora")
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--assets", help="Defaults to pi05/assets (pi05/smoke_assets for smoke tests)")
    p.add_argument("--checkpoints", help="Defaults to pi05/checkpoints (pi05/smoke_checkpoints for smoke tests)")
    p.add_argument("--exp-name", default="yam")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fsdp-devices", type=int, default=1)
    p.add_argument("--steps", type=int, help="Total steps; defaults to 30000, or 2 with --smoke-test")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Use only exported *depth_test* fixtures, not actual training demonstrations")
    p.add_argument("--prompt", help="Override every episode instruction (single-task data only)")
    p.add_argument("--openpi-dir", type=Path, required=True, help="OpenPI checkout containing scripts/train.py")
    return p


def main():
    args = parser().parse_args()
    args.assets = args.assets or ("pi05/smoke_assets" if args.smoke_test else "pi05/assets")
    args.checkpoints = args.checkpoints or ("pi05/smoke_checkpoints" if args.smoke_test else "pi05/checkpoints")
    args.steps = args.steps if args.steps is not None else (2 if args.smoke_test else 30000)
    if min(args.horizon, args.batch_size, args.fsdp_devices, args.steps, args.save_interval) < 1 or args.workers < 0:
        raise ValueError("Counts must be positive; workers must be nonnegative")
    if not (args.openpi_dir / "scripts" / "train.py").is_file():
        raise FileNotFoundError(args.openpi_dir / "scripts" / "train.py")
    from pi05.dataset import ABCDataset
    episode_glob = "*depth_test*" if args.smoke_test else "*"
    dataset = ABCDataset(args.data_root / "train", args.horizon, args.prompt,
                         episode_glob=episode_glob, decode=False)
    if not args.smoke_test:
        dataset.require_training_data()
    print(f"Selected {len(dataset.episodes)} episodes / {len(dataset)} frames; smoke_test={args.smoke_test}")
    import openpi.training.data_loader as dl
    from pi05.training_config import make_config
    config = make_config(args.mode, args.horizon, args.assets, args.checkpoints, args.exp_name,
                         args.batch_size, args.fsdp_devices, args.steps, args.workers,
                         args.resume, args.save_interval)
    stats = config.assets_dirs / "yam_abc" / "norm_stats.json"
    if not stats.is_file():
        raise FileNotFoundError(f"Compute normalization statistics first: {stats}")
    provenance = stats.parent / "dataset.json"
    if not provenance.is_file() or json.loads(provenance.read_text()) != dataset.provenance():
        raise ValueError("Statistics do not match this dataset/horizon/prompt; recompute them on the server")
    manifest = config.checkpoint_dir.parent / f"{args.exp_name}.recipe.json"
    recipe = dict(mode=args.mode, horizon=args.horizon, assets=str(Path(args.assets).resolve()),
                  data_root=str(args.data_root.resolve()), prompt=args.prompt,
                  normalization_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
                  smoke_test=args.smoke_test,
                  action_representation="joint_delta_gripper_absolute", ema_decay=None)
    if args.resume and (not manifest.exists() or json.loads(manifest.read_text()) != recipe):
        raise ValueError("Resume requires the original, matching recipe")
    if manifest.exists() and not args.resume and config.checkpoint_dir.exists():
        raise FileExistsError(f"Use a new experiment name or --resume: {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(recipe, indent=2))
    original = dl.create_torch_dataset
    def factory(data_config, action_horizon, model_config):
        if data_config.repo_id == "yam_abc":
            return ABCDataset(args.data_root / "train", action_horizon, args.prompt, episode_glob=episode_glob)
        return original(data_config, action_horizon, model_config)
    script = args.openpi_dir / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("yam_openpi_train", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dl.create_torch_dataset = factory
    try:
        logging.basicConfig(level=logging.INFO)
        module.main(config)
    finally:
        dl.create_torch_dataset = original


if __name__ == "__main__":
    main()

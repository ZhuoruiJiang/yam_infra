"""Serve full or LoRA JAX checkpoints using the identical YAM transforms."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1", help="Bind address; defaults to the local lab computer")
    p.add_argument("--port", type=int, default=8000)
    warmup = p.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", action="store_true", help="Compile with a synthetic observation; no exported dataset needed")
    warmup.add_argument("--warmup-episode", type=Path, help="Exported episode used to compile before accepting clients")
    return p


def checkpoint_paths(checkpoint, recipe):
    checkpoint = Path(checkpoint).expanduser().resolve()
    recipe = Path(recipe).expanduser().resolve()
    for path in (checkpoint / "params", checkpoint / "assets" / "yam_abc" / "norm_stats.json", recipe):
        if not path.exists():
            raise FileNotFoundError(f"Missing local inference artifact: {path}")
    return checkpoint, recipe


def make_inference_config(checkpoint, recipe):
    from openpi.training.config import AssetsConfig
    from pi05.training_config import make_config
    # The recipe records absolute training-server paths. Inference must use
    # the normalization copied with this checkpoint, never those old paths.
    cfg = make_config(recipe["mode"], recipe["horizon"])
    return dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, assets=AssetsConfig(assets_dir=str(Path(checkpoint) / "assets"), asset_id="yam_abc")))


def warmup_observation(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Synthetic warmup requires a nonempty prompt in the recipe; use --warmup-episode instead")
    return {"state": np.zeros(14, dtype=np.float32), "prompt": prompt,
            "images": {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("top", "left", "right")}}


def main():
    a = parser().parse_args()
    checkpoint, recipe_path = checkpoint_paths(a.checkpoint, a.recipe)
    recipe = json.loads(recipe_path.read_text())
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = make_inference_config(checkpoint, recipe)
    policy = create_trained_policy(cfg, checkpoint, default_prompt=recipe.get("prompt"))
    if a.warmup:
        policy.infer(warmup_observation(recipe.get("prompt")))
        print("Warmup complete", flush=True)
    elif a.warmup_episode:
        from pi05.dataset import ABCDataset
        ds = ABCDataset(a.warmup_episode.parent, recipe["horizon"], recipe.get("prompt"),
                        episode_glob=a.warmup_episode.name)
        sample = ds[0]
        policy.infer({key: sample[key] for key in ("images", "state", "prompt")})
        print("Warmup complete", flush=True)
    WebsocketPolicyServer(policy=policy, host=a.host, port=a.port,
                          metadata=cfg.policy_metadata).serve_forever()


if __name__ == "__main__":
    main()

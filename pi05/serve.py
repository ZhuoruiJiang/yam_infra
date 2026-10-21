"""Serve full or LoRA JAX checkpoints using the identical YAM transforms."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--warmup-episode", type=Path, help="Exported episode used to compile before accepting clients")
    a = p.parse_args()
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from pi05.training_config import make_config
    recipe = json.loads(a.recipe.read_text())
    cfg = make_config(recipe["mode"], recipe["horizon"], recipe["assets"])
    policy = create_trained_policy(cfg, a.checkpoint, default_prompt=recipe.get("prompt"))
    if a.warmup_episode:
        from pi05.dataset import ABCDataset
        ds = ABCDataset(a.warmup_episode.parent, recipe["horizon"], recipe.get("prompt"),
                        episode_glob=a.warmup_episode.name)
        sample = ds[0]
        policy.infer({key: sample[key] for key in ("images", "state", "prompt")})
        print("Warmup complete")
    WebsocketPolicyServer(policy=policy, host=a.host, port=a.port,
                          metadata=cfg.policy_metadata).serve_forever()


if __name__ == "__main__":
    main()

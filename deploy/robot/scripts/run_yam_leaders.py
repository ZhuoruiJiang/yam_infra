"""Bring up YAM teaching-handle leaders without starting followers."""

import argparse
from dataclasses import replace

from deploy.robot.launch import ProcessSpec, launch
from deploy.robot.leaders.yam_leader_config import YAM_LEADERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument("--publish-actions", action="store_true")
    args = parser.parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    specs = []
    for side in sides:
        cfg = replace(YAM_LEADERS[side], publish_actions=args.publish_actions)
        specs.append(ProcessSpec(cfg.name, "deploy.robot.leaders.yam_leader", {"cfg": cfg}))
    raise SystemExit(launch(specs))


if __name__ == "__main__":
    main()

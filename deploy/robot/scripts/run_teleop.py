"""Bimanual GELLO teleop: leader + follower nodes for both arms."""

import argparse
import os

from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.specs import teleop_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch leader/follower pairs only")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.profile:
        os.environ["ROBOT_PROFILE"] = args.profile
    # Suppress leader/follower stdout (servo torque/offsets, motor_states)
    # unless verbose.
    quiet = not (args.verbose or os.environ.get("DEPLOY_VERBOSE"))
    config = get_i2rt_config()
    if args.side != "both":
        if args.side not in config.robots:
            parser.error(f"Profile has no {args.side} robot")
        config.robots = {args.side: config.robots[args.side]}
    raise SystemExit(launch.launch(teleop_specs(config, quiet=quiet)))


if __name__ == "__main__":
    main()

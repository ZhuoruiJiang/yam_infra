"""Start existing camera/follower nodes, without any leader or ABC policy."""
import argparse
from dataclasses import replace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--followers", action="store_true", help="Enable follower motors and calibration")
    a = p.parse_args()
    from deploy.robot.config import get_i2rt_config
    from deploy.robot.launch import launch
    from deploy.robot.specs import camera_specs, follower_specs
    cfg = get_i2rt_config()
    specs = camera_specs(cfg)
    for spec in specs:
        spec.kwargs["cfg"] = replace(spec.kwargs["cfg"], enable_depth=False)
    if a.followers:
        specs.extend(follower_specs(cfg, quiet=False))
    raise SystemExit(launch(specs))


if __name__ == "__main__":
    main()

# Local i2rt driver package

This folder contains the copied `i2rt` Python package from
https://github.com/i2rt-robotics/i2rt. Its packaging is adapted to the flat
layout used here: `robots`, `motor_drivers`, and `robot_models` are directly
under this directory rather than an additional nested `i2rt` directory.

From the `yam_infra` root on Linux, activate your environment and install:

```bash
source .venv/bin/activate
python -m pip install -e ./i2rt
python -m pip check
python -c "from i2rt.robots.get_robot import get_yam_robot; print('Driver import OK')"
```

This installs declared Python dependencies into the active environment.
The copied dependency list includes `ruckig==0.15.3`, which builds from
source. Upstream constrains its build backend to `scikit-build-core<0.10`
in `[tool.uv]`; pip does not apply uv-specific settings. For a ruckig build
error, use uv with this folder as its project so the constraint is applied,
or pass an equivalent pip build constraint with a supported pip version.

The root project's deploy extra still names ABC's older pinned i2rt fork.
Installing that extra afterward may replace this local installation.
This copied driver is newer than the original pin; building the package
does not verify live arm compatibility. Confirm imports and leader-only
diagnostics on the Linux workstation before running followers.

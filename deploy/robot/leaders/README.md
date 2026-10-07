# YAM teaching-handle bring-up

The workstation's udev rules map:

| Role | CAN interface | USB adapter serial |
| --- | --- | --- |
| Left leader | can_leader_l | 001D0054594E501820313332 |
| Right leader | can_leader_r | 003F0053594E501820313332 |
| Left follower | can_follower_l | 00540051594E501820313332 |
| Right follower | can_follower_r | 00560056594E501820313332 |

Udev names interfaces; it does not prove the bus is up or communication works.
On the Linux workstation, inspect `ip -details link show can_leader_l` and
confirm the configured bitrate matches the hardware (the pinned i2rt driver
defaults to 1,000,000 bit/s). Use the existing workstation CAN setup.

From `yam_infra`, in the environment with i2rt installed:

```bash
python -m deploy.robot.scripts.run_yam_leaders --side left
python -m deploy.robot.scripts.run_yam_leaders --side right
# After testing each separately:
python -m deploy.robot.scripts.run_yam_leaders --side both
```

These commands start gravity compensation on leaders and print readings;
they do not start followers or publish action commands by default. Support
the arm when stopping: closing the driver ends gravity compensation.

Check that the six joint values follow hand motion and that the trigger and
both buttons respond. The diagnostic output includes the proposed gripper
command. It must be 1 when released (open) and 0 when fully squeezed (closed).
The pinned passive encoder driver already normalizes the trigger. If its
direction differs on this hardware, set `trigger_open` and `trigger_closed`
in `yam_leader_config.py` to the observed endpoint values.

To override endpoints for an individual diagnostic node:

```bash
python -m deploy.robot.leaders.yam_leader --name leader_left \
  --channel can_leader_l --trigger-open 0 --trigger-closed 1
```

An explicit `--publish-actions` publishes seven values on
`leader_left_actions` or `leader_right_actions`, compatible with ABC's
follower command format. A running follower subscribed to that topic will
move. Buttons are included in metadata but do not enable/disable following
in the existing ABC follower. Validate diagnostics before enabling commands.

The paired teleop launcher still uses the original GELLO profiles. YAM
profile integration awaits confirmation of the follower gripper variant.
The follower module is unchanged.

Implementation targets i2rt commit
`852f6fff33970fe9b882c53e6b28c66172c97291`, pinned in `pyproject.toml`.
Live testing is required if the workstation uses another driver version.

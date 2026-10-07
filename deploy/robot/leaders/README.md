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

## Paired teleop

The `yam_infra` robot profile uses the CAN mapping above, YAM teaching
handles, and confirmed `linear_4310` follower grippers. It also stores the
three known D405 camera serials; teleop starts only the selected arm pair.

Stop standalone leader diagnostics first so two processes do not share
the same CAN bus. Check that the selected leader and follower CAN buses
are up at 1 Mbit/s. Start with the left pair:

```bash
python -m deploy.robot.scripts.run_teleop --profile yam_infra --side left --verbose
```

This command moves hardware: the follower gripper calibrates during driver
initialization, then the follower interpolates to the leader's first pose
over approximately two seconds. Start with both arms in nearby poses and
keep the movement path clear. Check small joint movements and trigger
open/close before expanding to the right pair with `--side right`.

The existing follower attempts to return its joints to zero on Ctrl+C,
then closes the driver. Its shutdown is not a stationary hold. The leader
loses gravity compensation on exit and should be supported. Handle buttons
remain diagnostic metadata; they do not pause the follower.

The follower module is unchanged. Profile base geometry and rollout
starting poses are placeholders from ABC defaults; measure those before
policy rollout. This profile is selected explicitly, leaving existing
GELLO station defaults available.

The leader was implemented against ABC's pinned i2rt API and its diagnostic
readings were subsequently verified on both workstation leaders using the
copied driver. Paired follower motion still requires live verification.

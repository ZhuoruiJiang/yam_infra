# Teleop recording controls

| Input | Idle | Recording |
| --- | --- | --- |
| Space, s, right white | Start | Save and stop |
| d, left white | No action | Discard and stop |
| a | No action | Advance stage |

Old teleop keys b/c/x/j are no longer recorder commands. Ctrl+C retains
the existing behavior: finish/save an active recording and shut down.
Keyboard input retains the existing one-second cooldown.

YAM white is handle index 1. The leader publishes a one-value uint8 state
on `<leader_name>_white_button` independently of action publishing. Yellow
(index 0) is not read for recording. The listener subscribes without
conflation, detects rising edges, and debounces them for 100 ms. Holding
a white button emits one command; an initially held button must first be
released. Left maps to d and right maps to s on `KeyListener_key_presses`.
The listener never opens a leader CAN bus.

The data-collection launcher automatically enables these subscriptions for
YAM leaders in the selected profile. Select `ROBOT_PROFILE=yam_infra`;
the recorder reloads the same profile in its process. Explicitly pass
`--foot_pedal_device ''` when not using a pedal. Pedal inputs, if used, need
to emit the new keys; old physical pedal a/b/c labels no longer match.

## Test buttons before recording

Stop other processes using the leader CAN buses. In one terminal:

```bash
python -m deploy.robot.scripts.run_yam_leaders --side both
```

In a second terminal (same environment, same user):

```bash
DEPLOY_VERBOSE=1 python -m deploy.robot.key_listeners.key_listener \
  --left-white-topic leader_left_white_button \
  --right-white-topic leader_right_white_button
```

Left white prints `key='d'`, right white prints `key='s'`. A held button
prints once; release and press again to get another event. Yellow prints
no recording event. This test starts no follower or recorder, but leaders
run gravity compensation; support them when stopping.

For a short integrated control test after checking the buttons:

```bash
ROBOT_PROFILE=yam_infra DEPLOY_POST_VIDEO=0 \
python -m deploy.robot.scripts.run_data_record \
  --collection_name controls_test --foot_pedal_device '' --verbose
```

The full launcher starts both arm pairs and cameras. It prompts for an
optional session tag. Use right white to start/save, left white to discard,
and keyboard a to advance stage. Existing follower startup/shutdown motion
still applies. This step retains JPEG RGB recording and does not yet save
depth. MP4 rendering is disabled above to keep the test focused on controls.

# Camera-only bring-up

Run from the `yam_infra` root on the camera host. The existing deployment
environment targets Linux. Camera-only dependencies are `numpy`, `tyro`,
`pyzmq`, `pyrealsense2`, and `opencv-python` (with GUI support for preview).

List cameras without starting streams:

```bash
python -m deploy.robot.cameras.list_realsense
```

Start one RealSense camera, without arms, recorder, or key listener:

```bash
python -m deploy.robot.cameras.realsense --name test_camera --camera-serial YOUR_SERIAL --preview
```

Omit `--camera-serial` if exactly one camera is connected. Defaults are
640x480 at 30 Hz, with both RGB and depth enabled. The selected camera must
support those color and depth modes. Close RealSense Viewer before running
the node so it does not hold the camera. Preview requires a desktop display;
omit `--preview` for a headless host. Press Q or Escape in the preview, or
Ctrl+C in the terminal, to stop.

The left pane is RGB, the right pane is native depth colorized over a fixed
0–0.5 meter range, suitable for the D405. Change it using
`--preview-max-depth-m`. Invalid depth pixels appear black. Streams are
not explicitly aligned or resampled by this node.

The node publishes RGB uint8 on `test_camera_rgb` and native depth uint16
on `test_camera_depth`. Optional `--rgb-socket` and `--depth-socket` override
these topics; use explicit TCP endpoints if needed for a Windows host,
since the existing default IPC transport targets Linux.

Depth in meters is `depth_value * depth_scale_m`. Messages include camera
serial, frameset number for pairing, per-stream frame number, device
timestamp in milliseconds, and timestamp domain. Transport timestamps
remain host monotonic publication times, not device acquisition times.

This step enables capture and preview only. The existing HDF5 recorder
still subscribes to RGB only; depth persistence is a separate next step.

Bring-up passes when the serial is detected, both panes update, depth
changes as an object moves, and quitting releases the camera so the node
can restart. Hardware streaming must be verified on the camera host.

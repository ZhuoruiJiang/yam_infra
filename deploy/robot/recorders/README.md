# Teleop recording schema and offline validation

RGB stays JPEG quality 90 in variable-length uint8 datasets under
`data/left`, `data/right`, and `data/top`. Robot streams retain the existing
numeric layout. RealSense depth is added under `data/depth_<camera>` as
`(N, H, W)` uint16 arrays, compressed losslessly with gzip level 1 and
shuffle, with one frame per chunk.

Depth dataset attributes:

- `depth_scale_m`: multiply an integer pixel by this to obtain meters.
- `invalid_value=0`: no depth measurement.
- `aligned_to_rgb=False`: native camera depth, not an RGB-aligned image.

The sensor's scale is read at runtime; do not assume all cameras use
millimeters. Using float32 meters would double uncompressed storage and
does not increase sensor precision.

`timestamps/depth_<camera>` stores host monotonic publication timestamps
in nanoseconds, matching the existing timestamp convention. Acquisition
timestamps in milliseconds, frame IDs, frameset IDs, serials, intrinsics,
and depth-to-color extrinsics are retained under
`camera_metadata/<camera>/<rgb|depth>`. Metadata attributes ending in
`_json` contain JSON strings. Device timestamps may belong to different
camera clocks and are not automatically synchronized across cameras.

Both RGB and depth must have arrived recently before recording starts.
Existing latest-sample subscriptions remain; RGB/depth can drop samples
independently. Pair by camera and frameset ID, not array index. The validator
reports unmatched IDs. This schema does not claim every camera frame was
recorded. Inference/DAgger recording remains unchanged.

## Validate without hardware

From the repository root, in an environment containing h5py and numpy:

```bash
python -m deploy.robot.scripts.validate_recording /path/to/episode.h5
# New episodes should have depth; require it explicitly:
python -m deploy.robot.scripts.validate_recording /path/to/episode.h5 --require-depth
```

JSON output lists stream shapes, errors, and warnings. Errors produce a
nonzero exit status. Older JPEG-only episodes are accepted with missing
depth warnings unless `--require-depth` is supplied. The validator checks
JPEG markers, not complete JPEG decoding; it does not assess physical depth
accuracy or measured recording performance.

Checks cover expected left/right/top streams, both robot stream layouts,
nonempty data, matching timestamp and metadata counts, monotonic host
timestamps, native depth dtype/scale, stage boundaries, and discarded
usability flags. Synthetic round-trip tests include zero and uint16's
maximum value to verify exact preservation.

## Live check when hardware is available

Pull the update and run the existing collection command. Record a short
episode, save it, then run the validator with `--require-depth`. Inspect
three nonempty depth datasets, their scales, and metadata. Check an object
at a known distance before treating measurements as calibrated. Check any
pairing warnings and recording performance under the three-camera workload.
Repeat discard and confirm depth is retained in the discarded episode.

Writer failures are now surfaced to stop collection rather than allowing a
failed HDF5 writer to report a successful save. Failed files can remain
incomplete at their original location.

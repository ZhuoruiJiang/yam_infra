# Depth export and playback

Run the existing exporter. Available depth is exported by default:

```powershell
python -m deploy.recording.export --input-pattern "data/teleop_h5/*.h5" --output-dir data/exports/session --compute-norm-stats --val-mode split --scene-task empty
```

Each episode with depth contains `depth.h5` beside its RGB MP4 and
`states_actions.bin`. Native uint16 frames retain their original resolution,
use lossless gzip compression, and preserve `depth_scale_m`: multiply a value
by that scale to obtain meters. Zero means missing depth. Camera attributes
and the original depth camera metadata are retained.

The sidecar contains `timeline_ns`, and for each camera `frames`, `valid`,
`source_indices`, and `timestamps_ns`. Frames use the latest depth sample at
or before each exported RGB timeline timestamp. A missing initial sample is
marked invalid and filled with zeros. Samples can repeat or be skipped; this
is timestamp synchronization, not exact RGB/depth frameset pairing. The
original HDF5 remains the source for all original samples and metadata.

The existing `viz_episode.py` automatically shows depth when the sidecar
exists, with the same play/pause and frame controls as RGB. The preview uses
a fixed 0–1 meter scale (near red, far blue, missing black), in labeled camera
order. Preview resizing and colors do not modify exported depth. Depth remains
in its native camera coordinates; it is not aligned to RGB.

For real recordings, `--scene-task empty` selects ABC's empty robot scene
without changing the recorded task prompt. Then run:

```powershell
python viz_episode.py --episode-dir "data/exports/session/val/EPISODE" --mode pose
```

Older recordings with no depth export and play normally without a depth panel.
Install `h5py` in the viewer environment to read depth. Normalization statistics
computed from these exports are for testing or training from scratch; reuse
checkpoint statistics for finetuning.

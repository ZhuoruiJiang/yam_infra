# π0.5 on YAM (RGB only)

ABC reads exported episodes; OpenPI handles π0.5 transforms, normalization,
training and serving. No LeRobot conversion, ABC model preprocessing, depth
loading or throughput benchmarking. Joint order is left joints 1–6, left
gripper, right joints 1–6, right gripper. Recorded units remain unchanged:
arm targets in radians, grippers in the YAM driver's normalized [0,1] space.
Top/left/right cameras map to base/left-wrist/right-wrist model slots.
Training uses joint deltas relative to the observation state and absolute
gripper targets; serving reverses this into absolute follower targets.

## Server setup (Linux, Python 3.11)

Use a **separate OpenPI environment**, not `uv sync` in `yam_infra`: their
Torch versions differ. Copy this repository and exported data to the server.
Set the paths below to your actual server locations.

```bash
export YAM_ROOT=/path/to/yam_infra
export OPENPI_ROOT=/path/to/openpi
git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
uv python pin 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv pip install --no-deps -r "$YAM_ROOT/pi05/requirements-reader.txt"
export PYTHONPATH="$YAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PY="$OPENPI_ROOT/.venv/bin/python"
cd "$YAM_ROOT"
```

Install FFmpeg shared libraries using your server's package/environment
manager if TorchCodec reports missing libraries. The adapter uses ABC's
decoder; on TorchCodec 0.3 it falls back to exact indexed decoding because
that version predates ABC's custom-frame-map optimization. Use the Python
executable above, not `uv run` in this repository (which selects ABC's env).
OpenPI installs LeRobot as an upstream dependency, but this adapter never
uses its dataset loader. Base weights download automatically on first train.

## Validate the small test recordings

```bash
"$PY" -m pi05.validate_dataset --data-root data/exports/depth_test
"$PY" -m pi05.validate_dataset --data-root data/exports/depth_test --openpi
"$PY" -m unittest discover -s pi05/tests -v
```

Only the four `*depth_test*` exports in `train/` are selected; older
`recording_test` exports are excluded. Checks include exact raw state/action
values, RGB camera mapping, midpoint and episode-end chunks. `--openpi`
also checks delta round trips, normalization, tokenization and padded model
shapes; the tokenizer may download an asset. `--numeric-only` skips RGB.
These recordings are **fixtures only**: ordinary training/statistics commands
reject test-labelled data. The explicit smoke-test commands below allow them
to exercise the training pipeline. They do not assess policy quality.

## Training smoke test on depth_test

After the setup and validation commands above, run:

```bash
export DATA_ROOT="$YAM_ROOT/data/exports/depth_test"
"$PY" -m pi05.compute_norm_stats --data-root "$DATA_ROOT" --mode lora --smoke-test
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PY" -m pi05.train --openpi-dir "$OPENPI_ROOT" \
  --data-root "$DATA_ROOT" --mode lora --smoke-test --exp-name depth_smoke_lora \
  --batch-size 1 --workers 0 --steps 2 --save-interval 1
```

Replace GPU 0 with a reserved, otherwise idle GPU. `--smoke-test` selects
only the four `*depth_test*` episodes (1,588 frames), excludes `recording_test`,
and isolates statistics/checkpoints under `pi05/smoke_assets/` and
`pi05/smoke_checkpoints/`. Only RGB and numeric robot values are loaded.
Use a new experiment name for a fresh retry after an unsuccessful start.
First-time weight downloads and JAX compilation can take several minutes.
Success means finite losses and a saved checkpoint under
`pi05/smoke_checkpoints/pi05_yam_lora/depth_smoke_lora/1/`.

To test full fine-tuning after LoRA works, reserve all eight GPUs and run:

```bash
"$PY" -m pi05.compute_norm_stats --data-root "$DATA_ROOT" --mode full --smoke-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PY" -m pi05.train --openpi-dir "$OPENPI_ROOT" \
  --data-root "$DATA_ROOT" --mode full --smoke-test --exp-name depth_smoke_full \
  --batch-size 8 --fsdp-devices 8 --workers 0 --steps 2 --save-interval 1
```

No `torchrun` is needed: one JAX process uses all visible GPUs and shards
the full model. The 24 GB A5000 memory fit is not yet measured. If LoRA
cannot fit at batch 1, use multiple reserved GPUs with model sharding
(for example four GPUs, `--batch-size 4 --fsdp-devices 4`). Do not deploy
these fixture-trained checkpoints on the robot.

## Train on your actual demonstrations

An export root should contain `train/<episode>/` and optionally `val/`.
Each episode needs `episode_metadata.json`, `states_actions.bin` and
`combined_camera-images-rgb.mp4`. Metadata must identify top/left/right
cameras, 30 Hz and a task instruction (`instruction`, `prompt_timeline`, or
`task_name`). `--prompt 'your task'` overrides instructions for a single-task
dataset; pass it to **both** statistics and training. Existing ABC
`norm_stats.json` is ignored. Validation episodes never enter statistics or
training; this initial trainer uses upstream training-loss logging only.

```bash
export DATA_ROOT=/path/to/actual_demonstrations

# LoRA: first reserve GPUs with sufficient free VRAM.
"$PY" -m pi05.compute_norm_stats --data-root "$DATA_ROOT" --mode lora
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PY" -m pi05.train --openpi-dir "$OPENPI_ROOT" \
  --data-root "$DATA_ROOT" --mode lora --exp-name yam_lora \
  --batch-size 8 --fsdp-devices 1

# Full fine-tuning: shard the model across all eight reserved A5000s.
"$PY" -m pi05.compute_norm_stats --data-root "$DATA_ROOT" --mode full
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PY" -m pi05.train --openpi-dir "$OPENPI_ROOT" \
  --data-root "$DATA_ROOT" --mode full --exp-name yam_full \
  --batch-size 8 --fsdp-devices 8
```

For a first training smoke test, add `--steps 2 --save-interval 1
--exp-name smoke_lora` (or `smoke_full`). This uses actual demonstrations,
not the fixtures. A5000 memory fit is unverified: LoRA can be tight on 24 GB;
reduce the single-GPU batch to 1 or shard LoRA too if necessary. Global batch
must be divisible by visible GPU count, and `fsdp-devices` must divide that
count. Reserve the GPUs first: the supplied snapshot showed other jobs on all
eight. No existing jobs are stopped by these commands.

Defaults: horizon 50 at 30 Hz, 30k steps, LR 5e-5 with 1k warmup then cosine
decay, checkpoint every 1k, W&B disabled, EMA disabled for both modes. LoRA
uses OpenPI's `gemma_2b_lora` / `gemma_300m_lora` and matching freeze filter.
Full mode has no frozen weights. Add `--resume` with the same arguments to
resume; `--steps` is the total target step count. Outputs:

```text
pi05/assets/pi05_yam_<mode>/yam_abc/norm_stats.json
pi05/checkpoints/pi05_yam_<mode>/<experiment>/<step>/
pi05/checkpoints/pi05_yam_<mode>/<experiment>.recipe.json
```

## Inference later

On a GPU host, use the same OpenPI environment. Supply an actual saved step
directory (not its `params/` child) and its recipe. Full and LoRA checkpoints
are served through the same loader; no separate adapter merge is required.
Keep the entire checkpoint, including its copied normalization assets.

```bash
"$PY" -m pi05.serve \
  --checkpoint pi05/checkpoints/pi05_yam_lora/yam_lora/STEP \
  --recipe pi05/checkpoints/pi05_yam_lora/yam_lora.recipe.json
```

The first request may compile slowly. Add `--warmup-episode
data/exports/depth_test/train/261009-1238-depth_test-e8d070851cb64f95` to
compile using a recorded observation before live motion; the live client
rejects late responses.
On the lab workstation use the existing ABC **deploy** environment and select
your actual calibrated 30 Hz `ROBOT_PROFILE`. In one terminal:

```bash
ROBOT_PROFILE=YOUR_PROFILE python -m pi05.hardware --followers
```

This starts RGB-only cameras and follower nodes, including motor calibration.
Do not run teleop leaders or another rollout simultaneously. In a second
terminal, with the same profile and environment:

```bash
ROBOT_PROFILE=YOUR_PROFILE python -m pi05.run_yam \
  --host GPU_HOST --prompt 'your task' --chunks 1
# After inspecting commands-disabled results, enable follower commands:
ROBOT_PROFILE=YOUR_PROFILE python -m pi05.run_yam \
  --host GPU_HOST --prompt 'your task' --execute
```

Without `--execute`, real camera/state observations go to the model but no
action publishers are created. Live mode executes 10 actions then replans,
with finite-value, stale-camera/state, response-time and per-step motion
checks. Default limits: 0.1 rad/joint, 0.2/gripper; oversized targets stop
rather than being clipped. These are configurable checks, not collision
avoidance. Startup sends a hold at the measured pose, not a home move.
Ctrl+C stops the client; followers retain their last targets. Stop the
hardware launcher separately for its normal motor shutdown. Keep the lab's
physical stop available. Real motion and GPU training are not locally tested.

Sources: [OpenPI](https://github.com/Physical-Intelligence/openpi),
[TorchCodec versions](https://github.com/pytorch/torchcodec#installing-torchcodec).

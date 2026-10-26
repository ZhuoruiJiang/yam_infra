# π0.5 rollout on the lab computer

The lab computer now stores the checkpoint and runs OpenPI inference, cameras,
followers, and rollout. Cybertron5 is used only to copy the trained files.
Use three terminals on the **same Linux lab computer**. It needs an NVIDIA
GPU suitable for π0.5 inference. Keep OpenPI and the robot deployment in
separate virtual environments.

Run: `pick_and_place_cups_into_box_lora`, step `29999`.
Prompt: `put the cups into the box`.

## 1. Copy the checkpoint to the lab (once)

From the lab's `yam_infra` directory:

```bash
cd /path/to/yam_infra
export RUN=pick_and_place_cups_into_box_lora
export REMOTE_ROOT=/data2/zhuorui/yam_infra/pi05/checkpoints/pi05_yam_lora
export LOCAL_ROOT="$PWD/pi05/checkpoints/pi05_yam_lora"
mkdir -p "$LOCAL_ROOT/$RUN"

rsync -av "zhuorui@cybertron5.csres.utexas.edu:$REMOTE_ROOT/$RUN/29999/" \
  "$LOCAL_ROOT/$RUN/29999/"
rsync -av "zhuorui@cybertron5.csres.utexas.edu:$REMOTE_ROOT/$RUN.recipe.json" \
  "$LOCAL_ROOT/"
```

Wait until both transfers finish. Copy the whole numbered checkpoint,
including `params/` and `assets/`, and its recipe JSON. Keep them under
`pi05/checkpoints/`, which Git ignores. No exported data or training-server
statistics folder is needed. Serving loads statistics from the copied
checkpoint; old absolute paths in the recipe do not need editing.

## 2. Install OpenPI on the lab computer (once)

Skip this if the lab already has the matching OpenPI environment. These
commands place `openpi` beside the lab's `yam_infra` checkout and use the
same revision as training. `uv` and FFmpeg must already be available; no
sudo is needed by these commands.

```bash
cd /path/to/yam_infra
export OPENPI_ROOT="$(dirname "$PWD")/openpi"
git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
uv python pin 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

Leave the lab's working `yam_infra/.venv` deployment environment in place.
Only the local policy server uses `openpi/.venv`.

## 3. Lab terminal 1: load and warm up the local policy

```bash
cd /path/to/yam_infra
export PY="$(dirname "$PWD")/openpi/.venv/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="$PWD/pi05/checkpoints/pi05_yam_lora"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
"$PY" -u -m pi05.serve \
  --checkpoint "$CKPT_ROOT/pick_and_place_cups_into_box_lora/29999" \
  --recipe "$CKPT_ROOT/pick_and_place_cups_into_box_lora.recipe.json" \
  --warmup \
  --host 127.0.0.1 --port 8000
```

Select an available **lab GPU** (the example uses GPU 0). Wait for
`Warmup complete` and the server to start listening. `--warmup` compiles
inference with synthetic RGB/state inputs and the recipe's task prompt;
it does not move the arms or require a recorded episode. Keep this terminal
running. The server binds to localhost by default.

## 4. Lab terminal 2: start RGB cameras and followers

Use the lab's existing deployment environment:

```bash
cd /path/to/yam_infra
source .venv/bin/activate
export ROBOT_PROFILE=yam_infra

python -m pi05.hardware --followers
```

This enables follower motors and calibration. Keep teleop leaders and other
rollout processes stopped.

## 5. Lab terminal 3: test predictions, then enable motion

Use the same deployment environment as terminal 2:

```bash
cd /path/to/yam_infra
source .venv/bin/activate
export ROBOT_PROFILE=yam_infra

python -m pi05.run_yam \
  --host 127.0.0.1 --port 8000 \
  --prompt "put the cups into the box" \
  --chunks 1
```

`127.0.0.1` is the lab computer itself. No server connection or SSH tunnel
is used during rollout. This command requests one prediction without
sending predicted targets to the followers.

Once that succeeds, enable motion for a short trial:

```bash
python -m pi05.run_yam \
  --host 127.0.0.1 --port 8000 \
  --prompt "put the cups into the box" \
  --execute \
  --chunks 3 \
  --execute-steps 10
```

This executes 10 predicted targets per chunk at 30 Hz, then obtains a fresh
observation and replans. Keep the physical stop available; Ctrl+C stops
client commands but leaves followers holding their last target.

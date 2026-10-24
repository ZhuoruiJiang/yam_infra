# π0.5 rollout

Run in three terminals: one on the GPU server and two on the lab workstation.
This tests `pick_and_place_cups_into_box_lora`, trained with the instruction
`put the cups into the box`. Step `29999` is the final checkpoint of the
30,000-step run; use it after training has finished saving.

## 1. GPU server: load and warm up the policy

```bash
cd /data2/zhuorui/yam_infra

export PY=/data2/zhuorui/openpi/.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CKPT_ROOT="$PWD/pi05/checkpoints/pi05_yam_lora"

export STEP=29999
export EPISODE=/data2/zhuorui/yam_infra/data/exports/pick_and_place/train/261013-1832-pick_and_place-75a4eee59f3f4fa0

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
"$PY" -m pi05.serve \
  --checkpoint "$CKPT_ROOT/pick_and_place_cups_into_box_lora/$STEP" \
  --recipe "$CKPT_ROOT/pick_and_place_cups_into_box_lora.recipe.json" \
  --warmup-episode "$EPISODE" \
  --port 8000
```

Wait for `Warmup complete` and the server to start listening. Use a GPU available to you.

## 2. Lab terminal 1: start RGB cameras and followers

Run from the lab's `yam_infra` directory using its existing deployment environment:

```bash
cd /path/to/yam_infra
source .venv/bin/activate
export ROBOT_PROFILE=yam_infra

python -m pi05.hardware --followers
```

This enables follower motors and calibration. Keep teleop leaders and other rollout processes stopped.

## 3. Lab terminal 2: test predictions, then enable motion

Run from the lab's `yam_infra` directory using the same deployment environment:

```bash
cd /path/to/yam_infra
source .venv/bin/activate
export ROBOT_PROFILE=yam_infra

python -m pi05.run_yam \
  --host cybertron5.csres.utexas.edu \
  --port 8000 \
  --prompt "put the cups into the box" \
  --chunks 1
```

Use this exact prompt to match the retrained policy. Port 8000 on `cybertron5.csres.utexas.edu` must be accessible from the lab.

Once that succeeds, enable motion for a short trial:

```bash
python -m pi05.run_yam \
  --host cybertron5.csres.utexas.edu \
  --port 8000 \
  --prompt "put the cups into the box" \
  --execute \
  --chunks 3 \
  --execute-steps 10
```

This executes 10 predicted targets per chunk at 30 Hz, then obtains a fresh observation and replans.

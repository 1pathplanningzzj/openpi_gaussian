# RoboCasa Example

This directory contains the local RoboCasa evaluation client used with OpenPI policy servers.

The setup that works best in this repo is:

- use the OpenPI `uv_venv` environment for the policy server
- use a separate `robocasa` conda environment for the RoboCasa / robosuite client rollout

## Setup

### Prerequisites

- Python 3.10
- Conda
- RoboCasa kitchen assets

### Create the `robocasa` conda environment

```bash
conda create -n robocasa python=3.10
conda activate robocasa
cd /home/zijianzhang/openpi
```

### Install client-side dependencies

Using the Tsinghua mirror:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r examples/robocasa/requirements.in
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e third_party/robocasa
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e packages/openpi-client
```

If you want mp4 rollout videos:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "imageio[ffmpeg]"
```

### Download RoboCasa kitchen assets

```bash
python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py
```

### MuJoCo runtime setup

If `import mujoco` works but RoboCasa later fails with
`libmujoco.so.3.6.0: cannot open shared object file`, export the MuJoCo library path before running the client:

```bash
export LD_LIBRARY_PATH=/home/zijianzhang/miniconda3/envs/robocasa/lib/python3.10/site-packages/mujoco:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
```

Quick check:

```bash
python -c "import mujoco, robocasa, robosuite, openpi_client, tyro; print('all ok')"
```

## Environment sanity check

```bash
python examples/robocasa/check_env.py
```

## Running evaluation

### 1. Start the policy server

Run this in the OpenPI environment, not the `robocasa` conda environment:

```bash
cd /home/zijianzhang/openpi

CUDA_VISIBLE_DEVICES=2 /home/zijianzhang/openpi/uv_venv/bin/python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config pi05_robocasa \
  --policy.dir /data/zijianzhang/train_ckpts/pi05_robocasa/gaussian_world_model_robocasa_exp0411_singlegpu_bs4_aw0/2000
```

### 2. Run the RoboCasa client rollout

Run this in the `robocasa` conda environment:

```bash
conda activate robocasa
cd /home/zijianzhang/openpi

export LD_LIBRARY_PATH=/home/zijianzhang/miniconda3/envs/robocasa/lib/python3.10/site-packages/mujoco:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

python examples/robocasa/main.py \
  --host 127.0.0.1 \
  --port 8010 \
  --env-name PnPCounterToCab \
  --prompt "pick and place from counter to cabinet" \
  --num-episodes 20 \
  --max-steps 500 \
  --replan-steps 1
```

This evaluates the checkpoint using success rate:

- per-episode success / failure
- running success rate
- final success rate

### 3. Save rollout videos

To save videos, add the boolean flag `--save-videos`:

```bash
python examples/robocasa/main.py \
  --host 127.0.0.1 \
  --port 8010 \
  --env-name PnPCounterToCab \
  --prompt "pick and place from counter to cabinet" \
  --num-episodes 20 \
  --max-steps 500 \
  --replan-steps 1 \
  --save-videos \
  --video-dir /home/zijianzhang/openpi/robocasa_eval_videos_2000
```

Notes:

- do not write `--save-videos True`; this script uses a boolean flag style
- videos are only saved if an `imageio` backend such as `ffmpeg` is installed

## Expected outputs

During evaluation you should see lines like:

```text
Episode 1 result | success=False reward=0.0000 running_success_rate=0/1=0.000
Episode 2 result | success=True reward=0.0000 running_success_rate=1/2=0.500
Final success rate for PnPCounterToCab: 7/20 = 0.350
```

To extract the final metric from a log file:

```bash
grep "Final success rate" robocasa_eval.log
```

## Common issues

- `ModuleNotFoundError: No module named 'tyro'`
  - install `examples/robocasa/requirements.in` into the `robocasa` conda env
- `libmujoco.so.3.6.0: cannot open shared object file`
  - export `LD_LIBRARY_PATH` to the MuJoCo package directory as shown above
- `Environment PnPCounterToCab not found`
  - make sure `third_party/robocasa` is installed in the current env
- `Could not find a backend to open ... mp4`
  - install `imageio[ffmpeg]` or run without `--save-videos`

## Docker

An older Docker-based flow still exists in this directory:

```bash
docker compose build
docker compose run --rm robocasa
```

But for local checkpoint evaluation, the conda + OpenPI server split above is the recommended path.

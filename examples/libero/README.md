# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
uv run --active examples/libero/verify_spatial_alignment.py
# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
xvfb-run -a python examples/libero/main.py

```
<!-- uv venv ~/openpi/uv_venv --python 3.11 # 创建虚拟环境 uv 3.11 
source ~/openpi/uv_venv/bin/activate  # 激活虚拟环境
uv pip install -e .              #安装pi相关的一些库 在openpi的根目录下 
CUDA_VISIBLE_DEVICES=2 uv run --active scripts/serve_policy.py --env LIBERO
uv run --active scripts/serve_policy.py --env LIBERO  # uv 系统会有一个默认的 --active 这个是必须的 指定当前这个环境

uv run --active scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero_low_mem_finetune --policy.dir=/data1/zhangzj26/pi0_model/checkpoints/pi0_libero_low_mem_finetune/my_experiment/1000
Terminal window 2: -->
```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85


## 空间对齐假设验证实验

运行空间对齐分析实验来验证 2D 图像导致 3D 空间信息不准确的假设：

```bash
# 步骤 1: 运行评估收集数据（启用 3D guard 但不过载 action）
python examples/libero/main.py \
    --args.use-3d-guard \
    --args.no-active-3d-takeover \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 50 \
    --args.csv-filename spatial_alignment_analysis.csv

# 步骤 2: 分析结果
python examples/libero/analyze_spatial_hypothesis.py \
    --csv data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv

#  步骤 3: 对比实验 - 启用 3D takeover
python examples/libero/main.py \
    --args.use-3d-guard \
    --args.active-3d-takeover \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 50 \
    --args.csv-filename with_3d_takeover.csv
```

注意：tyro CLI 需要使用 `--args.` 前缀来访问嵌套参数。

## 有关渲染的配置

# 1. 更新源（确保能下载到对应版本）
sudo apt update

# 2. 补装所有缺失的核心依赖（和能运行服务器对齐）
sudo apt install -y \
  xvfb \
  libegl-dev \
  libegl-mesa0 \
  libegl1 \
  libgl1-mesa-dev \
  libglu1-mesa \
  libglu1-mesa-dev \
  libglx-dev \
  mesa-common-dev \
  mesa-utils \
  mesa-utils-bin

# 对于无头服务器，如果 EGL 不可用，可以安装 OSMesa：
sudo apt install -y libosmesa6-dev



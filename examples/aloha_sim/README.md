# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py

xvfb-run -a python examples/aloha_sim/main.py

```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM

uv run --active scripts/serve_policy.py --env ALOHA_SIM  # uv 系统会有一个默认的 --active 这个是必须的 指定当前这个环境
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_sim \
    --policy.dir=/data/zijianzhang/aloha_sim_checkpoint/pi0_aloha_sim/ 
```

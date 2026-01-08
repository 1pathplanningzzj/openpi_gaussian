# RoboCasa Example

This directory contains examples for running RoboCasa with OpenPI.

## Setup (Local with uv or conda)

You can set up the environment locally using `uv` or `conda`.

### Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) (optional, recommended) or Conda

### Option 1: Using uv

1. Create a virtual environment:
   ```bash
   uv venv .venv --python 3.10
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r examples/robocasa/requirements.in
   ```

3. Install `robocasa` in editable mode:
   ```bash
   uv pip install -e third_party/robocasa
   ```

4. (Optional) Install `openpi-client`:
   ```bash
   uv pip install -e packages/openpi-client
   ```

### Option 2: Using Conda

1. Create a Conda environment:
   ```bash
   conda create -n robocasa python=3.10
   conda activate robocasa
   ```

2. Install dependencies:
   ```bash
   pip install -r examples/robocasa/requirements.in
   ```

3. Install `robocasa` in editable mode:
   ```bash
   pip install -e third_party/robocasa
   ```

4. (Optional) Install `openpi-client`:
   ```bash
   pip install -e packages/openpi-client
   ```

## Running

1. Download kitchen assets (if not already done):
   ```bash
   python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py
   ```

2. Run the environment check script:
   ```bash
   python examples/robocasa/check_env.py
   ```

## Docker Setup (Alternative)

1. Build the Docker container:
   ```bash
   docker compose build
   ```

2. Run the container:
   ```bash
   docker compose run --rm robocasa
   ```

3. Inside the container, you may need to download assets:
   ```bash
   python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py
   ```

4. You can then run the check environment script:
   ```bash
   python examples/robocasa/check_env.py
   ```
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py     policy:checkpoint     --policy.config=pi0_robocasa     --policy.dir=checkpoints/pi0_robocasa/robocasa_test_01/2000
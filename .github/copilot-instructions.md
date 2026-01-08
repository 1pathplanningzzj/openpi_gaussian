# OpenPI AI Coding Agent Instructions

## Project Overview
OpenPI is a repository for open-source robotics models (π₀, π₀-FAST, π₀.₅) by Physical Intelligence. It provides VLA (Vision-Language-Action) models for various robot platforms.

### Architecture & Key Components
- **Core Library (`src/openpi`)**: The main codebase containing model definitions, training logic, and policy implementations. It primarily uses **JAX** (Flax/NNX) but also supports **PyTorch**.
- **Client Library (`packages/openpi-client`)**: A lightweight client for interacting with the models.
- **Examples (`examples/`)**: Contains integration code for specific environments (Aloha, Droid, Libero, Robocasa). Each example typically has its own `README.md` and setup instructions.
- **Scripts (`scripts/`)**: Entry points for training (`train.py`), serving (`serve_policy.py`), and data processing.
- **Third Party (`third_party/`)**: Contains vendored dependencies or submodules (e.g., `aloha`, `libero`, `robocasa`). **Do not modify files in `third_party/` unless necessary for integration fixes.**

## Tech Stack & Conventions
- **Language**: Python 3.11+
- **ML Frameworks**: 
  - **JAX** (Flax, NNX) is the primary framework for the core models.
  - **PyTorch** is used for some models and interoperability.
- **Dependency Management**: **`uv`** is the required tool. Do not use `pip` directly unless inside a `uv` managed venv.
- **Code Style**: Enforced by `ruff`.
- **Data Format**: The project heavily relies on **LeRobot** dataset formats (see `convert_*_to_lerobot.py` scripts).

## Developer Workflows

### Environment Setup
Always use `uv` for environment management:
```bash
uv sync
uv pip install -e .
# For specific examples, install their requirements:
uv pip install -r examples/<example_name>/requirements.in
```

### Build & Test
- **Linting & Formatting**:
  ```bash
  ruff check .
  ruff format .
  ```
- **Testing**:
  ```bash
  pytest
  ```
- **Pre-commit**: Ensure `pre-commit` is installed and run before pushing.

### Training & Inference
- **Training**: The main entry point is `scripts/train.py`. Configuration is handled via `tyro` and `openpi.training.config`.
  ```bash
  python scripts/train.py --config-name <config_name>
  ```
- **Inference**: Use `scripts/serve_policy.py` or the example notebooks.
- **Checkpoints**: Models are downloaded from `gs://openpi-assets` and cached in `~/.cache/openpi` (or `OPENPI_DATA_HOME`).

## Key Files & Directories
- `pyproject.toml`: Project dependencies and configuration.
- `src/openpi/training/config.py`: Training configuration definitions.
- `src/openpi/models/`: Model architectures (JAX/Flax).
- `src/openpi/policies/`: Policy implementations.
- `examples/*/convert_*_to_lerobot.py`: Data conversion scripts are critical for adapting new datasets.

## Common Patterns
- **JAX/Flax Usage**: The codebase uses `flax.nnx` for neural network modules. Familiarize yourself with NNX patterns.
- **Config System**: Configurations are often defined as dataclasses and parsed using `tyro`.
- **Data Loading**: Data loaders typically expect LeRobot formatted datasets.

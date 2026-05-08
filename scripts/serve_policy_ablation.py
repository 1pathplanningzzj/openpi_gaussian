"""Serve a PyTorch OpenPI policy with GS-VLA ablation-time config overrides.

This wrapper keeps evaluation model construction aligned with checkpoints trained
through scripts/run_libero_ablation.py, while leaving scripts/serve_policy.py
unchanged.
"""

from __future__ import annotations

import argparse
import logging
import socket
from typing import Any

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


ABLATION_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "no_structured_future_rollout": {
        "use_velocity_future_gaussians": False,
        "flow_loss_weight": 0.0,
    },
    "no_flow": {
        "flow_loss_weight": 0.0,
    },
    "no_future_depth_aux": {
        "use_future_depth_aux": False,
        "future_depth_aux_loss_weight": 0.0,
    },
    "no_renderer": {
        "render_loss_weight": 0.0,
        "current_frame_recon_loss_weight": 0.0,
        "use_lpips": False,
    },
    "no_depth": {
        "depth_loss_weight": 0.0,
        "use_future_depth_aux": False,
        "future_depth_aux_loss_weight": 0.0,
        "future_motion_depth_weight": 0.0,
    },
    "depth_only_proxy": {
        "use_velocity_future_gaussians": False,
        "render_loss_weight": 0.0,
        "current_frame_recon_loss_weight": 0.0,
        "flow_loss_weight": 0.0,
        "use_lpips": False,
    },
}


def _apply_ablation_preset(train_config: _config.TrainConfig, preset: str | None) -> _config.TrainConfig:
    if not preset:
        return train_config
    if preset not in ABLATION_MODEL_OVERRIDES:
        raise ValueError(f"Unknown ablation preset {preset!r}. Available: {sorted(ABLATION_MODEL_OVERRIDES)}")

    logging.info("Applying GS-VLA eval ablation preset: %s", preset)
    for name, value in ABLATION_MODEL_OVERRIDES[preset].items():
        object.__setattr__(train_config.model, name, value)
        logging.info("  model.%s = %r", name, value)
    return train_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI policy with optional GS-VLA ablation overrides.")
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--ablation-preset", default=None, choices=[None, *ABLATION_MODEL_OVERRIDES.keys()])
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_config = _apply_ablation_preset(_config.get_config(args.policy_config), args.ablation_preset)
    policy = _policy_config.create_trained_policy(train_config, args.policy_dir, default_prompt=args.default_prompt)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
